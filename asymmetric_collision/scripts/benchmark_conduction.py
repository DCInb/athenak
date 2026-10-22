#!/usr/bin/env python3
"""Check uniform implicit conduction, including tabular states above the last knot.

Run after loading the compiler environment. Builds the existing analytic regression
problem from an isolated CPU build's objects; leaves its builtin executable intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

from audit_thermal import table_query
from benchmark_exchange import input_text


def build_regression(root, build):
    out = build / "conduction-validation"
    out.mkdir(exist_ok=True)
    unit = next(r for r in json.loads((build / "compile_commands.json").read_text())
                if r["file"].endswith("/src/pgen/pgen.cpp"))
    wrapper = out / "pgen_conduction.cpp"
    wrapper.write_text('#include "athena.hpp"\n#undef USER_PROBLEM_ENABLED\n'
                       '#define USER_PROBLEM_ENABLED 1\n#include "'
                       + str(root / "src/pgen/pgen.cpp") + '"\n')
    commands, objects = [], []
    for source in (wrapper, root / "inputs/tests/implicit_conduction.cpp"):
        obj = out / (source.stem + ".o")
        command = shlex.split(unit["command"])
        command[command.index("-o") + 1] = str(obj)
        command[command.index("-c") + 1] = str(source)
        subprocess.run(command, cwd=unit["directory"], check=True)
        objects.append(str(obj)); commands.append(command)
    command = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    command[command.index("CMakeFiles/athena.dir/pgen/pgen.cpp.o")] = objects[0]
    command.insert(command.index("-o"), objects[1])
    executable = out / "athena-conduction"
    command[command.index("-o") + 1] = str(executable)
    subprocess.run(command, cwd=build / "src", check=True)
    commands.append(command)
    (out / "provenance.json").write_text(json.dumps({"commands": commands,
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}, indent=2) + "\n")
    return executable


def run(executable, folder, deck):
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "input.athinput"
    path.write_text(deck)
    with (folder / "run.log").open("w") as log:
        subprocess.run([str(executable), "-i", str(path), "-d", str(folder)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    return (folder / "run.log").read_text()


def history(folder):
    path = next(folder.glob("*.hydro.hst"))
    names = re.findall(r"\[\d+\]=(\S+)", path.read_text().splitlines()[1])
    return dict(zip(names, np.loadtxt(path).T))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--query-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build, output, case, query = [p.resolve() for p in (args.build_directory,
        args.output_dir, args.run_directory, args.query_executable)]
    output.mkdir(parents=True, exist_ok=True)
    analytic = build_regression(root, build)
    results = []
    for name in ("implicit_conduction", "implicit_conduction_dirichlet"):
        log = run(analytic, output / name, (root / f"inputs/tests/{name}.athinput").read_text())
        line = next(s for s in log.splitlines() if "conduction regression:" in s)
        results.append({"name": name, "self_check": line})
        print(line, flush=True)

    executable = build / "src/athena"
    unit = 1.e16 * (13. / 9.) * 1.660538921e-24 / 1.3806488e-16 / 11604.518121550082
    # One hydro step, identical in each pair, followed by conduction in the enabled run.
    # The no-conduction control isolates the electron update from hydrodynamic work.
    cases = [("hot_CH", 12000., 24000., 1., "flash-extrapolate"),
             ("hot_mixture", 12000., 24000., .4, "flash-extrapolate"),
             ("bounded_CH", 4000., 8000., 1., "clamp")]
    for name, left, right, y, policy in cases:
        rho = .05 / 1.1
        a, b = table_query(query, case, [[rho, left/unit, left/unit, y],
                                        [rho, right/unit, right/unit, y]])
        deck = input_text(case, name, rho, a["ei_code"], a["ee_code"],
                          1., y, .0001, .0001, "constant", -1., 1.)
        deck = re.sub(r"(?m)^pr = .*", f"pr = {(2./3.)*rho*(b['ei_code']+b['ee_code']):.17g}", deck)
        deck = deck.replace("reconstruct = plm", "reconstruct = dc")
        deck = deck.replace("eos_table_bounds = flash-extrapolate", "eos_table_bounds = " + policy)
        run(executable, output / (name + "_control"), deck)
        deck = deck.replace("<materials>", "alpha_iso = .05\n"
            "conduction_integrator = implicit\nconduction_theta = 1.\n"
            "conduction_linear_tolerance = 1.e-11\n"
            "conduction_nonlinear_tolerance = 1.e-10\n"
            "conduction_max_nonlinear_iterations = 12\nconduction_report = true\n<materials>")
        log = run(executable, output / name, deck)
        baseline = np.loadtxt(sorted((output / (name + "_control") / "tab").glob("*.tab"))[-1])
        conducted = np.loadtxt(sorted((output / name / "tab").glob("*.tab"))[-1])
        # Columns: gid, i, x, ion energy, electron energy, Ti, Te, flags.
        ion_error = float(np.max(np.abs(conducted[:, 3] / baseline[:, 3] - 1)))
        assert ion_error < 1.e-10, ion_error
        assert conducted[:, 6].min() > baseline[:, 6].min()
        assert conducted[:, 6].max() < baseline[:, 6].max()
        hst = history(output / name)
        energy_error = float(np.max(np.abs(hst["tot-E"] / hst["tot-E"][0] - 1)))
        assert energy_error < 1.e-11, energy_error
        assert np.all(conducted[:, 3:7] > 0)
        results.append({"name": name, "bounds_policy": policy,
                        "final_Te_range_ev": (conducted[:, 6][[np.argmin(conducted[:, 6]),
                                                               np.argmax(conducted[:, 6])]] * unit).tolist(),
                        "ion_energy_relative_difference_from_control": ion_error,
                        "relative_total_energy_error": energy_error,
                        "solver": next(s for s in log.splitlines() if "implicit conduction:" in s)})
        print(json.dumps(results[-1]), flush=True)
    (output / "results.json").write_text(json.dumps({"executable_sha256":
        hashlib.sha256(executable.read_bytes()).hexdigest(), "cases": results}, indent=2) + "\n")
    print("All uniform conduction benchmarks passed.", flush=True)


if __name__ == "__main__":
    main()
