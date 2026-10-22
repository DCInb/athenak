#!/usr/bin/env python3
"""Isolate shock-created temperature separation in the run's initially cold helium."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_thermal import table_query
from benchmark_exchange import input_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--query-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-temperature-ev", type=float, default=1.0001276984045693)
    args = parser.parse_args()
    run, output, executable, query = [p.resolve() for p in (args.run_directory,
        args.output_dir, args.executable, args.query_executable)]
    output.mkdir(parents=True, exist_ok=False)
    temperature_unit = 1.e16*(13./9.)*1.660538921e-24/1.3806488e-16/11604.518121550082
    initial_temperature = args.initial_temperature_ev
    rho, temperature = 1.e-5/1.1, initial_temperature/temperature_unit
    state = table_query(query, run, [rho, temperature, temperature, 0.])[0]
    deck = input_text(run, "helium_collision", rho, state["ei_code"], state["ee_code"],
                      1., 0., 2.e-4, .02, "spitzer", -1., 1.)
    # 64 cells over 32 um, two blocks; two initially identical He streams at +/-300 km/s.
    deck = deck.replace("nx1 = 8", "nx1 = 64", 1).replace("nx1 = 8", "nx1 = 32", 1)
    deck = deck.replace("x1min = -0.5", "x1min = -0.016").replace("x1max = 0.5", "x1max = 0.016")
    deck = deck.replace("ul = 0.", "ul = 0.3").replace("ur = 0.", "ur = -0.3")
    (output / "input.athinput").write_text(deck)
    with (output / "run.log").open("w") as log:
        subprocess.run([str(executable), "-i", str(output / "input.athinput"),
                        "-d", str(output)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
    a, b = [np.loadtxt(p) for p in sorted((output / "tab").glob("*.tab"))]
    order = np.argsort(b[:, 2]); b = b[order]
    ti, te = b[:, 5]*temperature_unit, b[:, 6]*temperature_unit
    assert np.allclose(a[:, 5:7]*temperature_unit, initial_temperature, rtol=1.e-9)
    assert np.isfinite(b).all() and np.all(b[:, 3:7] > 0)
    center = np.abs(b[:, 2]) < .008
    peak = np.argmax(np.where(center, ti, -1))
    hst = next(output.glob("*.hydro.hst"))
    names = re.findall(r"\[\d+\]=(\S+)", hst.read_text().splitlines()[1])
    history = dict(zip(names, np.loadtxt(hst).T))
    energy_error = float(np.max(np.abs(history["tot-E"] / history["tot-E"][0] - 1)))
    # The second periodic interface expands initially floor-temperature gas. Its
    # EOS floor can add a small amount of energy; retain and report that error.
    assert energy_error < 1.e-5, energy_error
    result = {"description": "Periodic colliding He streams; radiation and conduction absent; Spitzer exchange active",
              "initial_Ti_Te_ev": initial_temperature, "initial_density_gcc": 1.e-5,
              "initial_mean_ionization": float(state["zbar"]), "time_ns": .02,
              "central_peak_Ti_ev": float(ti[peak]), "same_cell_Te_ev": float(te[peak]),
              "central_peak_ratio": float(ti[peak]/te[peak]),
              "relative_total_energy_error": energy_error,
              "energy_note": "Initially cold expanding gas can activate EOS floors at the other periodic interface",
              "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savetxt(output / "profile.csv", np.column_stack([b[:, 2]*1000, ti, te]),
               delimiter=",", header="x_um,Ti_ev,Te_ev", comments="")
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.semilogy(b[:, 2]*1000, ti, label="Ions"); ax.semilogy(b[:, 2]*1000, te, label="Electrons")
    ax.set(xlabel="x [µm]", ylabel="Temperature [eV]",
           title=f"He collision · initially {initial_temperature:.3g} eV · t = 0.02 ns")
    ax.grid(alpha=.2); ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"helium_heating.{suffix}", dpi=170)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
