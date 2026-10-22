#!/usr/bin/env python3
"""Reference-informed cone-source transport benchmark; not a laser PIC/burn model.

Strozzi et al., https://arxiv.org/abs/1205.1594 (Eqs. 1,3) and
https://arxiv.org/abs/1111.5089 (monoenergetic comparison).
Generate reproducible number-weighted sources, then run the unmodified executable
interface on a finite 3D beam in a uniform DT slab. See docs/FAST_ELECTRONS.md.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import time

import numpy as np
from scipy.integrate import quad

from fast_electron_bai_benchmark import Checkpoint, capture, run as execute

ROOT = Path(__file__).resolve().parents[2]
KEV_ERG = 1.602176634e-9
AMU = 1.660538921e-24  # matches the executable's Units constant
REST_MEV = 8.1871057880e-7/(1000*KEV_ERG)
RADIUS_UM = 18.0
INTENSITY_W_CM2 = 1.0e20
WAVELENGTH_UM = 0.527
TP_MEV = REST_MEV*(math.sqrt(1+INTENSITY_W_CM2*WAVELENGTH_UM**2/1.37e18)-1)
AREA_CM2 = math.pi*(RADIUS_UM*1e-4)**2*math.gamma(1.25)
POWER_ERG_S = 0.52*INTENSITY_W_CM2*AREA_CM2*1e7
RHO = 450.0
NE_PER_GRAM = 1/(2.5*AMU)
ENERGY_UNIT = 1e4  # 1 micron, 1 ps, 1e-12 gram
PRESSURE_UNIT = 1e16
SOURCES = ("mono_collimated", "pic_collimated", "pic_divergent")


def run(exe, directory, switches, timeout, launcher=()):
    result = execute(exe, directory, switches, timeout, launcher)
    counts = re.findall(r"packets=(\d+)", (directory/"run.log").read_text())
    result["peak_live_packets_at_macro_steps"] = max(map(int, counts), default=0)
    return result


def gauss(a, b, order):
    x, w = np.polynomial.legendre.leggauss(order)
    return a+(x+1)*(b-a)/2, w*(b-a)/2


def spectrum(eps):
    return np.exp(-eps/0.19)/eps+0.82*np.exp(-eps/1.3)


def source_rows(name, ne=24, nt=12, nphi=4):
    """Integrate dN/dE dE and dN/dOmega sin(theta)dtheta dphi separately.

    Log-energy nodes preserve the cold component without overpopulating the tail.
    Narrow angular sources integrate to 2*Delta (<1.6e-8 discarded number fraction).
    The PIC angular source is confined to the forward hemisphere, as in the paper.
    """
    if name not in SOURCES or min(ne, nt, nphi) < 1 or nphi % 4:
        raise ValueError("known source and positive orders; azimuth count divisible by four")
    if name.startswith("mono"):
        energies, ew = np.array([1.5]), np.array([1.0])
    else:
        logeps, weights = gauss(math.log(0.12), math.log(6.37), ne)
        eps = np.exp(logeps)
        energies, ew = TP_MEV*eps, weights*eps*spectrum(eps)
        ew /= ew.sum()
    delta = math.radians(90 if name.endswith("divergent") else 10)
    theta, tw = gauss(0, min(math.pi/2, 2*delta), nt)
    tw *= np.sin(theta)*np.exp(-(theta/delta)**4)
    tw /= tw.sum()
    rows = np.array([[1000*energy, math.degrees(angle), 360*(p+0.5)/nphi,
                      we*wt/nphi]
                     for energy, we in zip(energies, ew)
                     for angle, wt in zip(theta, tw) for p in range(nphi)])
    rows[:, 3] /= rows[:, 3].sum()
    return rows


def source_moments(rows):
    e, theta, phi, w = rows.T
    w = w/w.sum()
    return {"rows": len(rows), "number_weight_sum": float(w.sum()),
            "mean_energy_MeV": float(np.dot(w, e)/1000),
            "mean_theta_deg": float(np.dot(w, theta)),
            "mean_cos_theta": float(np.dot(w, np.cos(np.deg2rad(theta)))),
            "energy_fraction_below_2_MeV": float(np.dot(w*(e < 2000), e)/np.dot(w, e))}


def stopping_reference(rows, setup):
    """Independent, continuous cold-target straight-ray energy balance.

    For dE/ds=-C/beta^2, the range primitive is R(E)=E^2/[C(E+mc^2)].
    Invert it at the first box boundary, or deposit the residual cutoff energy.
    Only valid for the separate dilute, no-fields, no-scattering verification runs.
    """
    u, w = gauss(0, 1, setup.radial_order)
    radius = RADIUS_UM*1.5*np.sqrt(u)
    w *= np.exp(-(radius/RADIUS_UM)**8)
    w /= w.sum()
    energy, theta, phi, ew = rows.T
    energy = energy*KEV_ERG
    theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    direction = np.array([np.cos(theta), np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi)]).T
    rest = REST_MEV*1000*KEV_ERG
    coefficient = 4*math.pi*(2.8179403205e-13)**2*rest*RHO*NE_PER_GRAM*10
    cutoff = 10*KEV_ERG
    ranges = energy**2/(coefficient*(energy+rest))
    cutoff_range = cutoff**2/(coefficient*(cutoff+rest))
    deposited = 0.0
    for r, wr in zip(radius, w):
        for i in range(setup.spatial_azimuths):
            az = 2*math.pi*(i+0.5)/setup.spatial_azimuths
            origin = np.array([0, r*math.cos(az), r*math.sin(az)])
            boundary = np.where(direction > 0, [64, 32, 32], [0, -32, -32])
            with np.errstate(divide="ignore"):
                path = np.min((boundary-origin)/direction, axis=1)*1e-4
            remainder = np.maximum(0, coefficient*(ranges-path))
            exit_energy = 0.5*(remainder+np.sqrt(remainder**2+4*remainder*rest))
            exit_energy = np.where(path >= ranges-cutoff_range, 0, exit_energy)
            deposited += wr*np.dot(ew, energy-exit_energy)/setup.spatial_azimuths
    return float(deposited/np.dot(ew, energy))


@dataclass(frozen=True)
class Setup:
    nx: int = 16
    radial_order: int = 8
    spatial_azimuths: int = 8
    field_cfl: float = 0.25
    hydro_cfl: float = 0.03
    duration_ps: float = 1.0
    tlim_ps: float = 1.5
    eta_cm2_s: float = 1000.0
    seed: int = 12345
    collisions: bool = True
    scattering: bool = True
    electromagnetic: bool = True
    power_scale: float = 1.0
    max_packets: int = 100000000
    loss_fraction: float = 0.05


def input_text(name, distribution, setup, outputs=True):
    s = setup
    if s.nx < 8 or s.nx % 8:
        raise ValueError("nx must be a positive multiple of eight, at least eight")
    p = 2*RHO*NE_PER_GRAM*KEV_ERG/PRESSURE_UNIT  # Te=Ti=1 keV
    deck = {
        "job": {"basename": name},
        "units": {"length_cgs": 1e-4, "mass_cgs": 1e-12, "time_cgs": 1e-12,
                  "mu": 1.25},
        "mesh": {"nghost": 2}, "meshblock": {},
        "time": {"evolution": "dynamic", "integrator": "rk2",
                 "cfl_number": s.hydro_cfl, "initial_dt": 0.02,
                 "tlim": s.tlim_ps, "nlim": -1, "ndiag": 20,
                 "align_outputs": "false"},
        "mhd": {"eos": "ideal", "reconstruct": "plm", "rsolver": "hlle",
                "gamma": 5/3, "two_temperature": "true", "dual_energy": "true",
                "electron_heat_capacity_fraction": 0.5,
                "initial_electron_temperature_ratio": 1, "t_ei": -1},
        "fast_electrons": {
            "electromagnetic": str(s.electromagnetic).lower(),
            "magnetic_diffusivity_cm2_s": s.eta_cm2_s, "field_cfl": s.field_cfl,
            "max_field_substeps": 100000, "power_erg_s": POWER_ERG_S*s.power_scale,
            "start_time": 0, "end_time": s.duration_ps, "origin_x1": 0,
            "origin_x2": 0, "origin_x3": 0, "source_radius": RADIUS_UM,
            "source_cutoff": 1.5, "source_radial_power": 8,
            "source_radial_order": s.radial_order,
            "source_azimuthal_points": s.spatial_azimuths,
            "distribution_file": distribution, "packets_per_bin": 1,
            "electron_number_per_gram": NE_PER_GRAM, "zeff": 1,
            "coulomb_log": 10, "collisions": str(s.collisions).lower(),
            "scattering": str(s.scattering).lower(), "thermalization_kev": 10,
            "thermalization_factor": 3, "max_fractional_loss": s.loss_fraction,
            "max_scatter": 0.05, "max_segments": 256, "max_transport_waves": 10000,
            "max_packets": s.max_packets, "periodic_transport": "false",
            "report_diagnostics": "true", "seed": s.seed},
        "problem": {"pgen_name": "shock_tube", "shock_dir": 1, "xshock": 32},
    }
    for d in range(3):
        deck["mesh"].update({f"nx{d+1}": s.nx, f"x{d+1}min": 0 if d == 0 else -32,
                             f"x{d+1}max": 64 if d == 0 else 32,
                             f"ix{d+1}_bc": "outflow", f"ox{d+1}_bc": "outflow"})
        # Eight blocks at every resolution; each MPI rank owns half the beam spot.
        deck["meshblock"][f"nx{d+1}"] = s.nx//2
    for side in ("l", "r"):
        deck["problem"].update({"d"+side: RHO, "p"+side: p})
        deck["problem"].update({k+side: 0 for k in ("u", "v", "w", "bx", "by", "bz")})
    if outputs:
        deck["output1"] = {"file_type": "rst", "dt": s.tlim_ps,
                           "single_file_per_rank": "false"}
    header = ("# Uniform DT transport adaptation of Strozzi cone-source models.\n"
              "# Point of injection represents the source plane, not a simulated cone.\n"
              "# 1 micron / 1 ps / 1e-12 g; Te=Ti=1 keV; constant eta; no burn.\n")
    return header+"\n".join(f"<{b}>\n"+"\n".join(f"{k} = {v}" for k, v in a.items())
                              +"\n" for b, a in deck.items())


def generate(output, ne=24, nt=12, nphi=4, setup=Setup()):
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"references": ["https://arxiv.org/abs/1205.1594",
                                "https://arxiv.org/abs/1111.5089"],
                "definition": "two controls and a divergent PIC-informed source; analytic reconstructions",
                "original_PSC_source": {"geometry": "planar preplasma; source used in cone-guided transport",
                                        "laser_intensity_W_cm2": 1.37e20,
                                        "wavelength_um": 1.0, "extraction_time_fs": 360},
                "adaptations": ["ponderomotive scaling to the transport laser parameters",
                                f"{setup.duration_ps:g} ps top-hat instead of the paper's 19 ps pulse",
                                "uniform 450 g/cm^3 DT slab, initially 1 keV",
                                "constant scalar magnetic diffusivity; no laser, cone material or burn"],
                "laser_intensity_W_cm2": INTENSITY_W_CM2, "wavelength_um": WAVELENGTH_UM,
                "ponderomotive_MeV": TP_MEV, "conversion_fraction": 0.52,
                "effective_area_cm2": AREA_CM2,
                "electron_power_W": POWER_ERG_S*setup.power_scale/1e7,
                "source_energy_J": POWER_ERG_S*setup.power_scale*setup.duration_ps*1e-19,
                "orders": {"energy": ne, "theta": nt, "phi": nphi},
                "setup": asdict(setup), "sources": {}}
    for name in SOURCES:
        rows = source_rows(name, ne, nt, nphi)
        path = output/f"{name}.dat"
        np.savetxt(path, rows, fmt="%.16e", header=(
            "Strozzi et al. 2012/2013 analytic source reconstruction.\n"
            "Not raw PIC samples. Collimated sources are artificial controls.\n"
            "kinetic_energy_keV theta_deg phi_deg relative_NUMBER_weight"))
        metadata["sources"][name] = source_moments(rows)
        metadata["sources"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output/f"{name}.athinput").write_text(input_text(name, path.name, setup))
    radius = np.linspace(0, 1.5*RADIUS_UM, 257)
    intensity = np.exp(-(radius/RADIUS_UM)**8)
    np.savetxt(output/"radial_profile.csv", np.column_stack([
        radius, intensity, .52*INTENSITY_W_CM2*setup.power_scale*intensity,
        2*math.pi*radius*intensity/(AREA_CM2*1e8)]), delimiter=",",
        header="radius_um,relative_intensity,electron_intensity_W_cm2,radial_number_pdf_per_um",
        comments="", fmt="%.16e")
    theta = np.linspace(0, math.pi/2, 721)
    angular = [np.rad2deg(theta)]
    for delta in [math.pi/18, math.pi/2]:
        angular.append(np.sin(theta)*np.exp(-(theta/delta)**4)/quad(
            lambda t: math.sin(t)*math.exp(-(t/delta)**4), 0, math.pi/2)[0]*math.pi/180)
    np.savetxt(output/"angular_profiles.csv", np.column_stack(angular), delimiter=",",
        header="theta_deg,collimated_number_pdf_per_degree,divergent_number_pdf_per_degree",
        comments="", fmt="%.16e")
    metadata["temporal_envelope"] = {"type": "top-hat", "start_ps": 0,
                                     "end_ps": setup.duration_ps,
                                     "electron_power_W": POWER_ERG_S*setup.power_scale/1e7}
    (output/"profiles.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def analyze(directory, setup):
    files = sorted((directory/"rst").glob("*.rst"))
    if len(files) != 2:
        raise ValueError(f"Expected initial/final checkpoints: {files}")
    cp = Checkpoint(files[-1])
    initial = Checkpoint(files[0])
    for gid in range(initial.nmb):
        u, _ = initial.fluid(gid)
        u = u[(slice(None), *initial.active)]
        for component in (5, 6):
            initial_keV = (2/3)*u[component]*PRESSURE_UNIT/(u[0]*NE_PER_GRAM*KEV_ERG)
            np.testing.assert_allclose(initial_keV, 1.0, rtol=1e-12, atol=0)
    n = setup.nx
    cube, ee, bmag, density = (np.zeros((n, n, n)) for _ in range(4))
    max_nf = 0.0
    max_divb = 0.0
    for gid, loc in enumerate(cp.logical):
        nmb = n//2
        i, j, k = loc[:3]*nmb
        sl = np.s_[k:k+nmb, j:j+nmb, i:i+nmb]
        data = cp.cell_data[gid][(slice(None), *cp.active)]
        u, b = cp.fluid(gid)
        ua = u[(slice(None), *cp.active)]
        cube[sl], ee[sl], density[sl] = data[1], ua[6], ua[0]
        max_nf = max(max_nf, float(np.max(data[2]/(ua[0]*NE_PER_GRAM))))
        bc = [0.5*(a.take(range(a.shape[2-d]-1), axis=2-d)
                   +a.take(range(1, a.shape[2-d]), axis=2-d))[cp.active]
              for d, a in enumerate(b)]
        bmag[sl] = np.sqrt(sum(a*a for a in bc))
        divb = sum(np.diff(a, axis=2-d)[cp.active]/(64/n)
                   for d, a in enumerate(b))
        max_divb = max(max_divb, float(np.max(np.abs(divb))))
    dx = 64/n
    x = (np.arange(n)+0.5)*dx
    y = x-32
    volume = dx**3
    deposited = cube.sum()*volume*ENERGY_UNIT
    axial = cube.sum(axis=(0, 1))*volume*ENERGY_UNIT/1e7/dx
    # Fractional geometric volumes avoid a resolution-dependent hotspot radius/depth.
    sub = ((np.arange(64)+0.5)/64-0.5)*dx
    yz = np.zeros((n, n))
    for a in sub:
        for b in sub:
            yz += ((y[:, None]+a)**2+(y[None, :]+b)**2 <= RADIUS_UM**2)/len(sub)**2
    depth = 1.2/RHO*1e4  # paper's optimal hot-spot areal depth, in microns
    fx = np.clip((depth-(x-dx/2))/dx, 0, 1)
    hotspot = np.sum(cube*yz[:, :, None]*fx)*volume*ENERGY_UNIT
    temperature = (2/3)*ee*PRESSURE_UNIT/(density*NE_PER_GRAM*KEV_ERG)
    if (not all(np.isfinite(a).all() for a in (cube, temperature, density, bmag))
            or np.any(density <= 0) or np.any(temperature <= 0)):
        raise ValueError("Nonfinite or nonpositive final plasma state")
    np.savez_compressed(directory/"fields.npz", x_um=x, y_um=y,
                        deposition_J_per_um=axial,
                        deposition_J_per_um3=cube*ENERGY_UNIT/1e7,
                        electron_temperature_keV=temperature,
                        magnetic_field_kT=bmag*math.sqrt(4*math.pi*PRESSURE_UNIT)/1e7)
    energy = POWER_ERG_S*setup.power_scale*setup.duration_ps*1e-12
    divb_normalized = max_divb*dx/max(float(bmag.max()), 1e-100)
    if divb_normalized > 1e-9:
        raise ValueError(f"Magnetic divergence exceeded tolerance: {divb_normalized}")
    return {"mesh_deposited_fraction": deposited/energy, "hotspot_fraction": hotspot/energy,
            "deposition_centroid_um": float(np.dot(x, axial)/axial.sum()),
            "max_Te_keV": float(temperature.max()),
            "max_B_kT": float(bmag.max()*math.sqrt(4*math.pi*PRESSURE_UNIT)/1e7),
            "final_max_nfast_over_ne": max_nf,
            "max_divB_dx_over_B": divb_normalized,
            "total_emitted_macro_packets": int(cp.header[6])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--select", default=".")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--launcher", default="mpiexec --oversubscribe --bind-to none -n {ranks}")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--ne", type=int, default=24)
    parser.add_argument("--nt", type=int, default=12)
    parser.add_argument("--nphi", type=int, default=4)
    for field, value in asdict(Setup()).items():
        if isinstance(value, bool):
            parser.add_argument("--no-"+field.replace("_", "-"), dest=field,
                                action="store_false", default=value)
        else:
            parser.add_argument("--"+field.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    setup = Setup(**{k: getattr(args, k) for k in asdict(Setup())})
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"metadata.json").exists():
        parser.error("choose a new output directory")
    profiles = args.profiles.resolve() if args.profiles else out/"profiles"
    if not args.profiles:
        generate(profiles, args.ne, args.nt, args.nphi, setup)
    if not args.exe:
        print((profiles/"profiles.json").read_text())
        return
    exe = args.exe.resolve()
    metadata = {"setup": asdict(setup), "exe": str(exe),
                "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "hostname": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "gpu": capture(["nvidia-smi", "-q"]),
                "mpi": capture(["mpiexec", "--version"]),
                "cpu": capture(["lscpu"]), "profiles": str(profiles),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT/"src/fast_electrons").glob("*.*")},
                "profile_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in profiles.glob("*.dat")},
                "timing_excludes_initialization": True,
                "timing_excludes_mesh_outputs": True, "ranks": args.ranks}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    names = [s for s in SOURCES if re.search(args.select, s)]
    if not names or args.repeats < 1:
        parser.error("select cases and at least one repeat")
    records = []
    with (out/"measurements.jsonl").open("w") as stream:
        for repeat in range(args.repeats):
            for name in names[repeat % len(names):]+names[:repeat % len(names)]:
                for ranks in args.ranks[::1 if repeat % 2 == 0 else -1]:
                    directory = out/f"{name}-r{ranks}-{repeat}"
                    distribution = profiles/f"{name}.dat"
                    deck = out/f"{name}.athinput"
                    deck.write_text(input_text(name, distribution, setup, False))
                    launch = shlex.split(args.launcher.format(ranks=ranks)) if ranks > 1 else []
                    result = run(exe, directory, ["-i", deck], args.timeout, launch)
                    ledger = result["ledger"]
                    injected = POWER_ERG_S*setup.power_scale*setup.duration_ps*1e-12
                    if (not math.isclose(result["time"], setup.tlim_ps, rel_tol=2e-6) or
                        not math.isclose(ledger["injected_erg"], injected, rel_tol=1e-10) or
                        abs(ledger["residual_erg"])/injected > 1e-9):
                        raise RuntimeError(f"Incomplete or nonconservative run: {directory}")
                    result.update(name=name, ranks=ranks, repeat=repeat)
                    records.append(result)
                    stream.write(json.dumps(result)+"\n")
                    stream.flush()
                    print(json.dumps(result), flush=True)
    # Separate scientific outputs; these runs are never pooled into timing statistics.
    if not args.timing_only:
        for name in names:
            directory = out/f"{name}-science"
            deck = out/f"{name}-science.athinput"
            deck.write_text(input_text(name, profiles/f"{name}.dat", setup, True))
            result = run(exe, directory, ["-i", deck], args.timeout)
            result.update(analyze(directory, setup))
            injected = POWER_ERG_S*setup.power_scale*setup.duration_ps*1e-12
            ledger = result["ledger"]
            if (not math.isclose(result["time"], setup.tlim_ps, rel_tol=2e-6) or
                not math.isclose(ledger["injected_erg"], injected, rel_tol=1e-10) or
                abs(ledger["residual_erg"])/injected > 1e-9 or
                not math.isclose(result["mesh_deposited_fraction"],
                                 ledger["deposited_erg"]/injected,
                                 rel_tol=1e-10, abs_tol=1e-12)):
                raise RuntimeError(f"Unfinished or inconsistent scientific run: {directory}")
            (directory/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({"science": name, **result}), flush=True)
    summary = []
    for name in names:
        for ranks in args.ranks:
            subset = [r for r in records if r["name"] == name and r["ranks"] == ranks]
            ts = [r["loop_s"] for r in subset]
            ledger = subset[0]["ledger"]
            summary.append({"name": name, "ranks": ranks, "median_s": statistics.median(ts),
                            "min_s": min(ts), "max_s": max(ts),
                            "deposited_fraction": ledger["deposited_erg"]/ledger["injected_erg"],
                            "escaped_fraction": ledger["escaped_erg"]/ledger["injected_erg"],
                            "active_fraction": ledger["active_erg"]/ledger["injected_erg"]})
    with (out/"summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
