#!/usr/bin/env python3
"""Sun & Bai (2023), sections 5.1--5.3, adapted to the fast-electron module.

Reference: https://arxiv.org/html/2304.10568 . See docs/FAST_ELECTRONS.md
for differences from the paper. This benchmarks the existing solver executable:
volume particle populations are seeded through its versioned restart interface.
Only uniform, unrefined, double-precision, native-endian MHD checkpoints are used.
No checkpoint from a user's simulation is modified.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import struct
import subprocess
import time
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ME, QE, CLIGHT = 9.1093837139e-28, 4.80320471257e-10, 2.99792458e10
REST = 8.1871057880e-7  # exactly the solver's energy normalization, in erg
PACKET = np.dtype([("x", "f8", 3), ("n", "f8", 3), ("energy", "f8"),
                   ("weight", "f8"), ("rng", "u8")])
NUMBER = r"[-+0-9.eE]+"


@dataclass(frozen=True)
class Case:
    name: str
    level: int = 0
    c: float = 10.0
    ppc: int = 1
    cfl_override: Optional[float] = None

    @property
    def nx(self):
        return (32 * 2**self.level, 1, 1) if self.name == "bell" else (
            (32, 32, 32) if self.name == "gyration" else (8, 8, 8))

    @property
    def mb(self):
        return tuple(min(n, 16 if self.name != "oscillation" else 4) for n in self.nx)

    @property
    def length(self):
        return 500.0 if self.name == "gyration" else (1.0 if self.name == "bell" else 8.0)

    @property
    def omega(self):
        return 2.0 * np.pi * 1.0e-6 if self.name == "bell" else 1.0

    @property
    def loading(self):
        return {"gyration": 1.0e-12, "bell": 2.0e5, "oscillation": 1.5}[self.name]

    @property
    def gamma(self):
        if self.name == "gyration":
            return math.sqrt(1.0 + (10.0/self.c)**2)  # p_y/m = 10
        v = 10.0 if self.name == "bell" else 0.1
        return 1.0 / math.sqrt(1.0 - (v/self.c)**2)

    @property
    def frequency(self):
        return self.omega * (self.loading + 1.0/self.gamma)

    @property
    def tlim(self):
        return {"gyration": 50.0, "bell": 0.5,
                "oscillation": 4.0*np.pi/self.frequency}[self.name]

    @property
    def field_cfl(self):
        if self.cfl_override is not None:
            return self.cfl_override
        return 0.2 if self.name == "bell" else 0.2 / 2**self.level

    @property
    def mass_unit(self):
        return (self.omega*ME*CLIGHT/QE)**2/(4.0*np.pi)  # length unit = 1 cm

    @property
    def time_unit(self):
        return self.c/CLIGHT

    @property
    def energy_unit(self):
        return self.mass_unit/self.time_unit**2

    @property
    def volume(self):
        return self.length**3  # inactive dimensions also have this extent


def input_text(case):
    deck = {
        "job": {"basename": case.name},
        "units": {"length_cgs": 1, "mass_cgs": case.mass_unit,
                  "time_cgs": case.time_unit, "mu": 0.5},
        "mesh": {"nghost": 2}, "meshblock": {},
        "time": {"evolution": "dynamic", "integrator": "rk2", "cfl_number": 0.3,
                 "tlim": case.tlim, "nlim": 0, "ndiag": 100,
                 "align_outputs": "false"},
        "mhd": {"eos": "ideal", "reconstruct": "plm", "rsolver": "hlld",
                "gamma": 5.0/3.0, "two_temperature": "true", "dual_energy": "true",
                "electron_heat_capacity_fraction": 0.5,
                "initial_electron_temperature_ratio": 1, "t_ei": -1},
        "fast_electrons": {"electromagnetic": "true", "magnetic_diffusivity_cm2_s": 0,
                           "field_cfl": case.field_cfl, "power_erg_s": 0,
                           "energy_kev": 100, "collisions": "false",
                           "scattering": "false", "periodic_transport": "true",
                           "max_packets": 2000000, "report_diagnostics": "true"},
        "problem": {"pgen_name": "shock_tube", "shock_dir": 1,
                    "xshock": case.length/2},
        "output1": {"file_type": "rst", "dt": case.tlim/24,
                    "single_file_per_rank": "false"},
    }
    for d in range(3):
        deck["mesh"].update({f"nx{d+1}": case.nx[d], f"x{d+1}min": 0,
                             f"x{d+1}max": case.length,
                             f"ix{d+1}_bc": "periodic", f"ox{d+1}_bc": "periodic"})
        deck["meshblock"][f"nx{d+1}"] = case.mb[d]
    velocity = [0., 0., 0.]
    magnetic = [-1., 0., 0.]
    if case.name == "gyration":
        velocity[2] = 1.0
    elif case.name == "oscillation":
        velocity[1] = -case.loading * case.gamma * 0.1
        magnetic = [0., 0., 1.]
    for side in ("l", "r"):
        deck["problem"].update({"d"+side: 1, "p"+side: 1})
        deck["problem"].update({k+side: v for k, v in zip(("u", "v", "w"), velocity)})
        deck["problem"].update({k+side: v for k, v in zip(("bx", "by", "bz"), magnetic)})
    return "\n".join(f"<{block}>\n" + "\n".join(f"{k} = {v}" for k, v in values.items())
                     + "\n" for block, values in deck.items())


class Checkpoint:
    """Checked reader for this benchmark's native MHD + FEPIC001 fixture format."""

    def __init__(self, path):
        self.raw = bytearray(Path(path).read_bytes())
        raw = self.raw
        self.par_end = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
        off = self.par_end
        self.nmb, self.root_level = struct.unpack_from("=2i", raw, off)
        off += 8
        self.region = np.frombuffer(raw, "=f8", 9, off).copy()
        off += 72
        self.mesh_ind = np.frombuffer(raw, "=i4", 19, off).copy()
        off += 76
        self.ind = np.frombuffer(raw, "=i4", 19, off).copy()
        off += 76
        self.time, self.dt, self.cycle = struct.unpack_from("=2di", raw, off)
        off += 20
        self.logical = np.frombuffer(raw, "=i4", self.nmb*4, off).reshape(-1, 4).copy()
        off += self.nmb*20  # logical locations followed by float costs
        self.block_bytes, = struct.unpack_from("=Q", raw, off)
        self.fluid_offset = off + 8
        ng, nx, ny, nz = map(int, self.ind[:4])
        self.shape = (nz+2*ng if nz > 1 else 1, ny+2*ng if ny > 1 else 1, nx+2*ng)
        self.cells = math.prod(self.shape)
        self.active = (slice(ng, ng+nz) if nz > 1 else slice(0, 1),
                       slice(ng, ng+ny) if ny > 1 else slice(0, 1), slice(ng, ng+nx))
        nk, nj, ni = self.shape
        self.bshapes = [(nk, nj, ni+1), (nk, nj+1, ni), (nk+1, nj, ni)]
        nu, rem = divmod(self.block_bytes//8-sum(math.prod(s) for s in self.bshapes),
                         self.cells)
        if rem or nu != 7 or self.block_bytes % 8:
            raise ValueError("Expected double MHD with two temperature scalars only")
        if np.any(self.logical[:, 3] != self.root_level):
            raise ValueError("Only uniform, unrefined benchmark checkpoints are supported")
        self.footer = self.fluid_offset + self.nmb*self.block_bytes
        if raw[self.footer:self.footer+8] != b"FEPIC001":
            raise ValueError("Unexpected native restart layout or missing particle footer")
        off = self.footer+8
        self.header = np.frombuffer(raw, "=u8", 13, off).copy()
        off += 104
        h = self.header
        if (h[0] not in (1, 2) or tuple(h[1:5]) != (8, 11, self.cells, self.nmb) or h[12] != PACKET.itemsize
                or tuple(h[8:12]) != (0, self.nmb, 0, 0)):
            raise ValueError("Incompatible particle footer")
        self.ledger = np.frombuffer(raw, "=f8", 2, off).copy()
        off += 16
        self.emission_schedule = bytes(raw[off:off+40]) if h[0] == 2 else b""
        off += len(self.emission_schedule)
        self.spectrum = bytes(raw[off:off+int(h[5])*40])
        off += len(self.spectrum)
        counts = np.frombuffer(raw, "=u8", self.nmb, off).copy()
        off += 8*self.nmb
        self.packets, self.cell_data = [], []
        for count in counts:
            self.cell_data.append(np.frombuffer(raw, "=f8", 11*self.cells, off).reshape(
                (11, *self.shape)))
            off += 88*self.cells
            self.packets.append(np.frombuffer(raw, PACKET, int(count), off))
            off += int(count)*PACKET.itemsize
        if off != len(raw):
            raise ValueError("Unexpected trailing bytes in benchmark checkpoint")

    def fluid(self, gid):
        off = self.fluid_offset+gid*self.block_bytes
        u = np.frombuffer(self.raw, "=f8", 7*self.cells, off).reshape((7, *self.shape))
        off += 56*self.cells
        b = []
        for shape in self.bshapes:
            b.append(np.frombuffer(self.raw, "=f8", math.prod(shape), off).reshape(shape))
            off += math.prod(shape)*8
        return u, b

    def block_origin(self, gid):
        return self.region[:3] + self.logical[gid, :3]*self.ind[1:4]*self.region[6:9]

    def all_packets(self):
        return np.sort(np.concatenate(self.packets), order="rng")


def seed_particles(template, target, case):
    cp = Checkpoint(template)
    if cp.header[0] != 1:
        raise ValueError("Synthetic initial populations require a legacy-emission template")
    ncell = math.prod(case.nx)
    count = ncell*case.ppc
    weight = case.loading*case.volume*case.mass_unit/(ME*count)
    energy = REST*(case.gamma-1.0)
    packets = []
    seq = 1
    ng = int(cp.ind[0])
    for gid in range(cp.nmb):
        origin = cp.block_origin(gid)
        dx = cp.region[6:9]
        if case.name == "bell":
            # Growing helical eigenmode, q < 0: reverse all B components from
            # the positive-charge convention. eps=v_A/v_CR=0.1, deltaB=1e-3.
            u, b = cp.fluid(gid)
            x = origin[0]+(np.arange(cp.shape[2])-ng+0.5)*dx[0]
            phase, a, eps = 2*np.pi*x, 1.0e-3, 0.1
            co, si, grow = np.cos(phase), np.sin(phase), math.sqrt(1-eps**2)
            u[1] = 0
            u[2] = -a*eps*co+a*grow*si
            u[3] = -a*grow*co-a*eps*si
            b[0][:] = -1
            b[1][:] = -a*co
            b[2][:] = -a*si
            u[4] = u[5]+u[6]+0.5*(u[1]**2+u[2]**2+u[3]**2)/u[0]+0.5*(1+a*a)
        # A quiet, regular particle lattice. For 3D ppc must be a cube.
        ndim = sum(n > 1 for n in case.nx)
        side = round(case.ppc**(1/ndim))
        if side**ndim != case.ppc:
            raise ValueError("Use a perfect-cube ppc in 3D")
        axes = [origin[d]+(np.arange(case.mb[d]*side)+0.5)*dx[d]/side
                if case.nx[d] > 1 else np.array([case.length/2]) for d in range(3)]
        xyz = np.meshgrid(*axes, indexing="ij")
        p = np.zeros(xyz[0].size, dtype=PACKET)
        p["x"] = np.stack([a.ravel() for a in xyz], axis=-1)
        p["n"][:, 0 if case.name == "bell" else 1] = 1
        p["energy"], p["weight"] = energy, weight
        p["rng"] = np.arange(seq, seq+len(p), dtype="u8")
        seq += len(p)
        packets.append(p)
    assert seq-1 == count
    h = cp.header.copy()
    h[6] = count
    with Path(target).open("wb") as f:
        f.write(cp.raw[:cp.footer])
        f.write(b"FEPIC001")
        f.write(h.tobytes())
        f.write(np.array([weight*energy*count, 0.], dtype="f8").tobytes())
        f.write(cp.emission_schedule)
        f.write(cp.spectrum)
        f.write(np.array([len(p) for p in packets], dtype="u8").tobytes())
        for p in packets:
            f.write(bytes(88*cp.cells))
            f.write(p.tobytes())
    # Round-trip validation checks the fixture before the executable consumes it.
    check = Checkpoint(target)
    assert len(check.all_packets()) == count
    assert np.isclose(check.ledger[0], sum(np.sum(p["weight"]*p["energy"])
                                         for p in check.packets), rtol=1e-14)


def capture(command):
    try:
        return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def run(exe, directory, switches, timeout, launcher=()):
    directory.mkdir(parents=True, exist_ok=False)
    command = [*launcher, str(exe), *map(str, switches)]
    if not launcher and hasattr(os, "sched_getaffinity"):
        command = ["taskset", "-c", str(min(os.sched_getaffinity(0))), *command]
    (directory/"command.json").write_text(json.dumps(command, indent=2))
    start = time.perf_counter()
    with (directory/"run.log").open("w") as log:
        p = subprocess.Popen(command, cwd=directory, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise RuntimeError(f"Run timeout: {directory}") from None
    wall = time.perf_counter()-start
    log = (directory/"run.log").read_text()
    if code:
        raise RuntimeError(f"Run failed ({code}): {directory}\n{log[-3500:]}")
    end = re.findall(r"^time=("+NUMBER+r") cycle=(\d+)$", log, re.M)[-1]
    diagnostics = [dict((k, float(v)) for k, v in re.findall(
        r"([a-z_]+)=("+NUMBER+r")", line)) for line in log.splitlines()
        if line.startswith("# fast_electrons ")]
    return {"wall_s": wall,
            "loop_s": float(re.search(r"cpu time used\s*=\s*("+NUMBER+r")", log)[1]),
            "time": float(end[0]), "cycles": int(end[1]),
            "ledger": diagnostics[-1] if diagnostics else {}}


def gyration_exact(t, c=10.0):
    """Equations (24)--(25); solve the monotonic implicit phase by Newton iteration."""
    t = np.asarray(t)
    v0, u0, omega = 10., 1., 1.
    a = math.sqrt(v0*v0+u0*u0)/(math.sqrt(1-u0*u0/c**2)*omega)
    phase0 = math.acos(math.sqrt((1-u0*u0/c**2)/(1+u0*u0/v0**2)))
    rate = omega*math.sqrt((1-u0*u0/c**2)/(1+v0*v0/c**2))
    phase = phase0+rate*t
    for _ in range(12):
        f = phase-phase0-rate*((1-u0*u0/c**2)*t+u0*v0/(c*c*omega))
        f += rate*u0*a/(c*c)*np.cos(phase)
        phase -= f/(1-rate*u0*a/(c*c)*np.sin(phase))
    y = (math.sqrt(v0*v0+u0*u0)*np.sin(phase)
         - math.sqrt(1+v0*v0/c**2)*u0)/((1-u0*u0/c**2)*omega)
    z = u0*t+a*np.cos(phase)-v0/omega
    py = math.sqrt((v0*v0+u0*u0)/(1-u0*u0/c**2))*np.cos(phase)
    pz = u0*math.sqrt(1+v0*v0/c**2)/(1-u0*u0/c**2)
    pz -= math.sqrt(v0*v0+u0*u0)/(1-u0*u0/c**2)*np.sin(phase)
    return np.stack([y, z], axis=-1), np.stack([py, pz], axis=-1)


def oscillation_exact(t, case):
    """Single electron species, zero total momentum, uniform B=+z.

    Gas u=-r p/m, so dp/dt=-Omega*(1/gamma+r)*p cross z. The
    momentum norm and gamma are constant in the continuum solution.
    """
    t = np.asarray(t)
    p0 = case.gamma*0.1
    return np.stack([-p0*np.sin(case.frequency*t), p0*np.cos(case.frequency*t)], axis=-1)


def snapshot(path, case, initial_position):
    cp = Checkpoint(path)
    particles = cp.all_packets()
    q = particles["energy"]/REST
    pm = case.c*np.sqrt(q*(q+2))[:, None]*particles["n"]  # p / m in code units
    mass_weights = particles["weight"]*ME/case.mass_unit
    volume = np.prod(cp.region[6:9])
    total_u, deposited, mode = np.zeros(7), 0., 0j
    gas_kinetic = 0.
    max_divb = 0.
    for gid in range(cp.nmb):
        u, b = cp.fluid(gid)
        ua = u[(slice(None), *cp.active)]
        total_u += np.sum(ua, axis=(1, 2, 3))*volume
        gas_kinetic += 0.5*np.sum(np.sum(ua[1:4]**2, axis=0)/ua[0])*volume
        deposited += np.sum(cp.cell_data[gid][(1, *cp.active)])*volume
        div = np.diff(b[0], axis=2)/cp.region[6]
        if case.nx[1] > 1:
            div += np.diff(b[1], axis=1)/cp.region[7]
        if case.nx[2] > 1:
            div += np.diff(b[2], axis=0)/cp.region[8]
        max_divb = max(max_divb, float(np.max(np.abs(div[cp.active]))))
        if case.name == "bell":
            x = cp.block_origin(gid)[0]+(np.arange(case.mb[0])+0.5)*cp.region[6]
            by = (0.5*(b[1][:, :-1, :]+b[1][:, 1:, :]))[cp.active].ravel()
            mode += 2*np.sum(by*np.exp(-2j*np.pi*x))*volume/case.volume
    particle_energy = np.sum(particles["weight"]*particles["energy"])/case.energy_unit
    particle_momentum = np.sum(mass_weights[:, None]*pm, axis=0)
    mean_pm = np.average(pm, axis=0, weights=mass_weights)
    mean_gamma = np.average(1+q, weights=mass_weights)
    displacement = (particles["x"][0]-initial_position+case.length/2) % case.length-case.length/2
    return {"time": cp.time, "count": len(particles),
            "gas_velocity": (total_u[1:4]/total_u[0]).tolist(),
            "particle_p": mean_pm.tolist(),
            "comoving_gamma": float((mean_gamma-mean_pm[2]/case.c**2)
                                     /math.sqrt(1-1/case.c**2)) if case.name == "gyration" else None,
            "displacement": displacement.tolist(), "mode": [mode.real, mode.imag],
            "energy": float(total_u[4]+particle_energy),
            "kinetic_energy": float(gas_kinetic+particle_energy),
            "momentum": (total_u[1:4]+particle_momentum).tolist(),
            "ledger_rel_error": float((cp.ledger[0]/case.energy_unit-particle_energy-deposited)
                                       /(cp.ledger[0]/case.energy_unit)),
            "max_divb": max_divb}


def analyze(case, seed, directory):
    initial = Checkpoint(seed).all_packets()["x"][0]
    paths = [seed, *sorted((directory/"rst").glob("*.rst"))]
    rows = [snapshot(p, case, initial) for p in paths]
    # Terminal output can duplicate a scheduled snapshot.
    rows = list({r["time"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["time"])
    ts = np.array([r["time"] for r in rows])
    if not math.isclose(ts[-1], case.tlim, rel_tol=1e-12):
        raise RuntimeError(f"Incomplete trajectory: {directory}")
    if any(r["count"] != math.prod(case.nx)*case.ppc for r in rows):
        raise RuntimeError(f"Population changed: {directory}")
    metrics = {"case": case.name, "level": case.level, "c": case.c,
               "cells": math.prod(case.nx), "packets": rows[0]["count"],
               "field_cfl": case.field_cfl,
               "max_ledger_rel_error": max(abs(r["ledger_rel_error"]) for r in rows),
               "max_total_energy_rel_error": max(abs(r["energy"]-rows[0]["energy"])
                                                  /abs(rows[0]["energy"]) for r in rows),
               "max_divb": max(r["max_divb"] for r in rows)}
    if case.name == "gyration":
        x, p = gyration_exact(ts, case.c)
        observed_x = np.array([r["displacement"][1:] for r in rows])
        observed_p = np.array([r["particle_p"][1:] for r in rows])
        metrics["max_position_error_over_rg"] = float(np.max(np.linalg.norm(observed_x-x, axis=1))/10)
        metrics["max_momentum_rel_error"] = float(np.max(np.linalg.norm(observed_p-p, axis=1))/10)
        metrics["max_comoving_energy_rel_error"] = max(abs(r["comoving_gamma"]/rows[0]["comoving_gamma"]-1)
                                                       for r in rows)
    elif case.name == "bell":
        mode = np.array([complex(*r["mode"]) for r in rows])
        growth = 2*np.pi*math.sqrt(0.99)
        frequency = 2*np.pi*0.1
        # Fit the entire linear interval; report both imaginary and real frequency.
        fitted_growth = np.polyfit(ts, np.log(np.abs(mode)), 1)[0]
        fitted_frequency = -np.polyfit(ts, np.unwrap(np.angle(mode)), 1)[0]
        metrics.update(growth=float(fitted_growth), analytic_growth=growth,
                       growth_rel_error=float(abs(fitted_growth/growth-1)),
                       frequency=float(fitted_frequency), analytic_frequency=frequency,
                       frequency_rel_error=float(abs(fitted_frequency/frequency-1)))
    else:
        p0 = case.gamma*0.1
        exact = oscillation_exact(ts, case)
        actual = np.array([r["particle_p"][:2] for r in rows])
        gas = np.array([r["gas_velocity"][:2] for r in rows])
        phase = np.unwrap(np.arctan2(-actual[:, 0], actual[:, 1]))
        frequency = np.polyfit(ts, phase, 1)[0]
        metrics.update(analytic_frequency=case.frequency, frequency=float(frequency),
                       frequency_rel_error=float(abs(frequency/case.frequency-1)),
                       max_particle_momentum_rel_error=float(np.max(np.linalg.norm(actual-exact, axis=1))/p0),
                       max_gas_velocity_rel_error=float(np.max(np.linalg.norm(gas+case.loading*exact, axis=1))
                                                        /(case.loading*p0)),
                       max_kinetic_energy_rel_error=max(abs(r["kinetic_energy"]/rows[0]["kinetic_energy"]-1)
                                                       for r in rows),
                       max_momentum_error_over_initial_particle_momentum=float(max(
                           np.linalg.norm(np.array(r["momentum"])-rows[0]["momentum"]) for r in rows)
                           /(case.loading*case.volume*p0)))
    if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))):
        raise RuntimeError(f"Nonfinite benchmark result: {directory}")
    (directory/"trajectory.json").write_text(json.dumps(rows, indent=2))
    (directory/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT,
                        help="Source tree matching --exe, for provenance")
    parser.add_argument("--select", default="gyration|bell|oscillation")
    parser.add_argument("--levels", default="0,1,2", help="Accuracy refinement levels")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-level", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="mpiexec --oversubscribe -n {ranks}")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--bell-c", type=float, default=100)
    parser.add_argument("--field-cfl", type=float,
                        help="Override field CFL, e.g. to hold substeps fixed while changing C")
    args = parser.parse_args()
    args.exe, args.output = args.exe.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.repeats < 0 or args.ranks < 1:
        parser.error("repeats must be nonnegative and ranks must be positive")
    if args.field_cfl is not None and not 0 < args.field_cfl <= 0.25:
        parser.error("field-cfl must be in (0, 0.25]")
    launcher = shlex.split(args.launcher.format(ranks=args.ranks)) if args.ranks > 1 else []
    source_root = args.source_dir.resolve()
    source_paths = sorted((source_root/"src/fast_electrons").glob("*"))
    if not source_paths:
        parser.error("source-dir must contain the matching src/fast_electrons directory")
    metadata = {"reference": "https://arxiv.org/html/2304.10568", "host": platform.node(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job": os.environ.get("SLURM_JOB_ID"), "ranks": args.ranks,
                "executable": str(args.exe), "sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                "argv": os.sys.argv, "cpu": capture(["lscpu"]),
                "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]),
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_dir": str(source_root),
                "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_paths if p.is_file()}}
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    metrics, timings, summaries = [], [], []
    levels = [int(n) for n in args.levels.split(",") if n]
    cases = [Case("gyration"), Case("bell", c=args.bell_c, ppc=2), Case("oscillation", ppc=64)]
    for base in cases:
        if not re.search(args.select, base.name):
            continue
        base = replace(base, cfl_override=args.field_cfl)
        fixtures = {}
        for level in sorted(set(levels + ([args.timing_level] if args.repeats else []))):
            case = replace(base, level=level)
            case_dir = args.output/f"{case.name}-L{level}"
            case_dir.mkdir()
            deck = case_dir/"input.athinput"
            deck.write_text(input_text(case))
            (case_dir/"case.json").write_text(json.dumps(asdict(case), indent=2))
            run(args.exe, case_dir/"template", ["-i", deck], args.timeout)
            seed = case_dir/"initial.rst"
            seed_particles(sorted((case_dir/"template/rst").glob("*.rst"))[-1], seed, case)
            fixtures[level] = seed
            if level in levels:
                directory = case_dir/"accuracy"
                result = run(args.exe, directory, ["-r", seed, "time/nlim=-1",
                             "output1/file_number=0", "output1/last_time=0"], args.timeout, launcher)
                metric = analyze(case, seed, directory)
                metric["accuracy_run_wall_s"] = result["wall_s"]
                metrics.append(metric)
                write_csv(args.output/"accuracy.csv", metrics)
                print(json.dumps(metric), flush=True)
        if not args.repeats:
            continue
        case = replace(base, level=args.timing_level)
        seed = fixtures[case.level]
        case_times = []
        # A separate warmup and output-free evolution: AthenaK's timer otherwise
        # includes scheduled AND terminal checkpoint writes.
        for repeat in range(-1, args.repeats):
            directory = args.output/f"{case.name}-timing"/("warmup" if repeat < 0 else str(repeat))
            result = run(args.exe, directory, ["-r", seed, "time/nlim=-1", "output1/dt=-1"],
                         args.timeout, launcher)
            if not math.isclose(result["time"], case.tlim, rel_tol=2e-6):
                raise RuntimeError(f"Incomplete timing run: {directory}")
            if result["cycles"] <= 0 or result["ledger"]["packets"] != math.prod(case.nx)*case.ppc:
                raise RuntimeError(f"Invalid timing workload: {directory}")
            if abs(result["ledger"]["residual_erg"]/result["ledger"]["injected_erg"]) > 1e-8:
                raise RuntimeError(f"Energy ledger failed: {directory}")
            row = {"case": case.name, "level": case.level, "repeat": repeat, "ranks": args.ranks,
                   "cells": math.prod(case.nx), "packets": math.prod(case.nx)*case.ppc,
                   **{k: v for k, v in result.items() if k != "ledger"},
                   "ledger_rel_error": result["ledger"]["residual_erg"]/result["ledger"]["injected_erg"]}
            if repeat >= 0:
                timings.append(row)
                case_times.append(row)
                write_csv(args.output/"timings.csv", timings)
            print(json.dumps({"timing": row}), flush=True)
        summary = {"case": case.name, "level": case.level, "ranks": args.ranks,
                   "cells": row["cells"], "packets": row["packets"], "repeats": args.repeats,
                   "median_loop_s": statistics.median(r["loop_s"] for r in case_times),
                   "min_loop_s": min(r["loop_s"] for r in case_times),
                   "max_loop_s": max(r["loop_s"] for r in case_times),
                   "median_wall_s": statistics.median(r["wall_s"] for r in case_times)}
        summaries.append(summary)
        write_csv(args.output/"performance.csv", summaries)
    (args.output/"complete.json").write_text(json.dumps({"accuracy": metrics, "performance": summaries}, indent=2))


if __name__ == "__main__":
    main()
