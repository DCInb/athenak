#!/usr/bin/env python3
"""Run homogeneous production exchange against analytic and conservative RK4 references."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_thermal import table_query


def input_text(run, basename, rho, ei, ee, ratio, y, dt, duration, mode, tau, multiplier):
    pressure = (2. / 3.) * rho * (ei + ee)
    return f"""<job>
basename = {basename}
<units>
length_cgs = 0.1
mass_cgs = 0.0011
time_cgs = 1.e-9
mu = 1.4444444444444444
<mesh>
nghost = 2
nx1 = 8
nx2 = 1
nx3 = 1
x1min = -0.5
x1max = 0.5
x2min = -0.5
x2max = 0.5
x3min = -0.5
x3max = 0.5
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 8
nx2 = 1
nx3 = 1
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.25
tlim = {duration:.17g}
nlim = -1
initial_dt = {dt:.17g}
align_outputs = true
ndiag = 100000
<hydro>
eos = ideal
reconstruct = plm
rsolver = llf
gamma = 1.6666666666666667
dfloor = 1.e-8
pfloor = 1.e-14
tfloor = 1.e-12
nscalars = 2
two_temperature = true
dual_energy = true
initial_electron_temperature_ratio = {ratio:.17g}
t_ei_model = {mode}
t_ei = {tau:.17g}
t_ei_coulomb_log = 10.
t_ei_spitzer_multiplier = {multiplier:.17g}
t_ei_temperature_floor_kelvin = 1.
<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
material0_zbar = 3.5
material0_zeff = 5.285714285714286
material1_name = He
material1_abar = 4.002602
material1_zbar = 2.
material1_zeff = 2.
material0_eos_table_file = {run / 'material_tables/ch.2t_eos'}
material1_eos_table_file = {run / 'material_tables/he.2t_eos'}
eos_mixing_rule = partial_density_additive
eos_table_bounds = flash-extrapolate
eos_table_interpolation = geometric
<problem>
pgen_name = shock_tube
shock_dir = 1
xshock = 0.
dl = {rho:.17g}
dr = {rho:.17g}
pl = {pressure:.17g}
pr = {pressure:.17g}
ul = 0.
vl = 0.
wl = 0.
ur = 0.
vr = 0.
wr = 0.
yl = {y:.17g}
yr = {y:.17g}
<output1>
file_type = hst
data_format = %24.16e
dt = {dt:.17g}
<output2>
file_type = tab
variable = hydro_2t
data_format = %24.16e
dt = {duration:.17g}
slice_x2 = 0.
slice_x3 = 0.
"""


def run_case(executable, query, run, output, name, physical, dt, duration,
             mode="spitzer", tau=-1., multiplier=1., reuse=False):
    folder = output / name
    rho_gcc, ti_ev, te_ev, y = physical
    temperature_unit = 1.e16 * (13. / 9.) * 1.660538921e-24 / 1.3806488e-16 / 11604.518121550082
    rho = rho_gcc / 1.1
    state = table_query(query, run, [rho, ti_ev / temperature_unit, te_ev / temperature_unit, y])[0]
    deck = input_text(run, name, rho, state["ei_code"], state["ee_code"],
                      te_ev / ti_ev, y, dt, duration, mode, tau, multiplier)
    elapsed = None
    if reuse and (folder / "temperatures.csv").exists():
        assert (folder / "input.athinput").read_text() == deck
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "input.athinput").write_text(deck)
        start = time.perf_counter()
        with (folder / "run.log").open("w") as log:
            subprocess.run([str(executable), "-i", str(folder / "input.athinput"),
                            "-d", str(folder)], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        elapsed = time.perf_counter() - start
    hst = next(folder.glob("*.hydro.hst"))
    names = re.findall(r"\[\d+\]=(\S+)", hst.read_text().splitlines()[1])
    raw = np.loadtxt(hst)
    columns = dict(zip(names, raw.T))
    energy_rows = np.column_stack([np.full(len(raw), rho),
                                  columns["scal-2"] / columns["mass"],
                                  columns["scal-3"] / columns["mass"], np.full(len(raw), y)])
    states = table_query(query, run, energy_rows, "energy")
    times = columns["time"]
    ti, te = states["ti_code"] * temperature_unit, states["te_code"] * temperature_unit
    measured = np.column_stack([times, ti, te, energy_rows[:, 1:3]])
    np.savetxt(folder / "temperatures.csv", measured, delimiter=",",
               header="time_ns,Ti_ev,Te_ev,ei_code,ee_code", comments="")
    initial_energy = energy_rows[0, 1:3].sum()
    energy_error = float(np.max(np.abs(energy_rows[:, 1:3].sum(axis=1) / initial_energy - 1)))
    assert energy_error < 2.e-10, energy_error
    assert np.all(ti > 0) and np.all(te > 0)
    assert np.allclose([ti[0], te[0]], [ti_ev, te_ev], rtol=3.e-10)
    assert abs(times[-1] - duration) < 2.e-12
    assert np.all(np.abs(np.diff(times) - dt) < 1.e-12)
    if mode == "constant" and tau < 0:
        assert np.allclose(ti, ti[0], rtol=2.e-10) and np.allclose(te, te[0], rtol=2.e-10)
    if mode == "constant" and tau == 0:
        assert np.allclose(ti[1:], te[1:], rtol=2.e-10)
    record = {"name": name, "rho_gcc": rho_gcc, "initial_Ti_ev": ti_ev,
              "initial_Te_ev": te_ev, "Y_CH": y, "dt_ns": dt, "duration_ns": duration,
              "mode": mode, "tau_constant_ns": tau, "tau_multiplier": multiplier,
              "initial_tau_ei_ns": float(state["tau_ei_ns"]),
              "initial_tau_difference_ns": float(state["tau_difference_ns"]),
              "final_Ti_ev": float(ti[-1]), "final_Te_ev": float(te[-1]),
              "relative_energy_error": energy_error, "wall_seconds": elapsed}
    print(json.dumps(record), flush=True)
    return record, measured, state


def reference(query, run, output, name, state, physical, duration, max_step):
    arguments = [state["rho_code"], state["ti_code"], state["te_code"],
                 physical[3], duration, max_step]
    command = [str(query), str(run / "material_tables/ch.2t_eos"),
               str(run / "material_tables/he.2t_eos"), "relax"] + [f"{v:.17g}" for v in arguments]
    path = output / f"{name}_reference.csv"
    with path.open("w") as stream:
        subprocess.run(command, stdout=stream, check=True, timeout=120)
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--query-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed cases after checking executable hashes and inputs")
    args = parser.parse_args()
    run, output, executable, query = [p.resolve() for p in
        (args.run_directory, args.output_dir, args.executable, args.query_executable)]
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                  "query_sha256": hashlib.sha256(query.read_bytes()).hexdigest()}
    provenance_path = output / "provenance.json"
    if args.resume:
        assert json.loads(provenance_path.read_text()) == provenance
    else:
        assert not provenance_path.exists(), "Use a fresh output directory or --resume"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    cases = [
        ("hot_He", [8.518279501e-5, 43219.97416, 206.27911, 1.22601866e-8], .0002, .5),
        ("mixed_corona", [3.2238329e-4, 4713.3234, 555.4257, .75146574], .0002, .5),
        ("dense_CH", [100., 1000., 100., 1.], 2.e-7, 2.e-5),
    ]
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, (name, physical, dt, duration) in zip(axes, cases):
        coarse, coarse_data, state = run_case(executable, query, run, output,
                                              name + "_coarse", physical, dt, duration,
                                              reuse=args.resume)
        fine, fine_data, _ = run_case(executable, query, run, output,
                                      name + "_fine", physical, dt / 4, duration,
                                      reuse=args.resume)
        ref = reference(query, run, output, name, state, physical, duration, dt / 16)
        ref_check = reference(query, run, output, name + "_refined", state, physical,
                              duration, dt / 32)
        reference_error = float(np.max(np.abs(ref[-1, 1:3] / ref_check[-1, 1:3] - 1)))
        # Piecewise table derivatives need not be smooth at interpolation knots.
        # Require both an absolute reference tolerance and separation from solver error.
        assert reference_error < 1.e-6, reference_error
        for record, data in ((coarse, coarse_data), (fine, fine_data)):
            record["reference_final_Ti_ev"] = float(ref[-1, 1])
            record["reference_final_Te_ev"] = float(ref[-1, 2])
            record["relative_temperature_error"] = float(np.max(np.abs(data[-1, 1:3] / ref[-1, 1:3] - 1)))
            record["reference_refinement_error"] = reference_error
            results.append(record)
        assert fine["relative_temperature_error"] <= coarse["relative_temperature_error"] * .4 + 1.e-9
        assert fine["relative_temperature_error"] < 5.e-4
        assert reference_error < .05 * fine["relative_temperature_error"] + 1.e-9
        xscale = 1.e6 if name == "dense_CH" else 1.
        unit = "fs" if name == "dense_CH" else "ns"
        for i, label, color in ((1, "Ti", "#b84920"), (2, "Te", "#334fbd")):
            ax.plot(ref[:, 0] * xscale, ref[:, i], color=color, label=label + " RK4 reference")
            stride = max(1, len(coarse_data) // 15)
            ax.plot(coarse_data[::stride, 0] * xscale, coarse_data[::stride, i], ".",
                    color=color, label=label + " production")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel(f"Time [{unit}]"); ax.set_ylabel("Temperature [eV]")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"exchange_benchmark.{suffix}", dpi=170)
    plt.close(fig)
    hot = cases[0]
    for name, tau in (("exchange_disabled", -1.), ("instant_equilibrium", 0.)):
        record, _, _ = run_case(executable, query, run, output, name, hot[1],
                                 .0002, .002, mode="constant", tau=tau, reuse=args.resume)
        results.append(record)
    summary = {"executable": str(executable),
               "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
               "reference": "Conservative RK4 with tau and EOS heat capacity updated at every stage",
               "cases": results}
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("All exchange benchmarks passed.", flush=True)


if __name__ == "__main__":
    main()
