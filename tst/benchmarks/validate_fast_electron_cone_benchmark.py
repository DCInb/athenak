#!/usr/bin/env python3
"""Independent stopping oracles and convergence runs for the cone-source benchmark.

Convergence jobs include outputs and are excluded from performance medians.
The mesh-only suite preserves source sampling while doubling mesh resolution.
The full refinement additionally refines the finite source surface.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex

import numpy as np

from fast_electron_cone_benchmark import (
    Setup, SOURCES, POWER_ERG_S, generate, input_text, run, analyze, stopping_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["oracles", "short", "mesh-only",
                                            "full-refinement", "full-pulse", "no-fields"],
                        required=True)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--variants", nargs="+",
                        choices=["mesh32", "field-time", "fluid-time", "spatial"],
                        help="Select independent full-pulse refinements (default: all)")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--launcher", default="mpiexec --oversubscribe --bind-to none -n 2")
    args = parser.parse_args()
    exe, output, profiles = args.exe.resolve(), args.output.resolve(), args.profiles.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = Setup()
    cases = []
    if args.suite == "oracles":
        for loss in [.05, .01]:
            setup = replace(base, electromagnetic=False, scattering=False,
                            power_scale=1e-6, duration_ps=.1, tlim_ps=.6,
                            loss_fraction=loss)
            cases += [(f"loss{loss}", name, setup, profiles) for name in SOURCES]
    elif args.suite == "short":
        setup = replace(base, duration_ps=.1, tlim_ps=.6)
        variants = {"coarse": setup,
                    "mesh": replace(setup, nx=32),
                    "time": replace(setup, field_cfl=.125, hydro_cfl=.015),
                    "spatial": replace(setup, radial_order=16, spatial_azimuths=16),
                    "seed": replace(setup, seed=12346)}
        for label, variant in variants.items():
            cases += [(label, name, variant, profiles) for name in SOURCES]
        high = output/"fine-profiles"
        generate(high, ne=48, nt=24, nphi=8, setup=setup)
        cases += [("spectrum", name, setup, high) for name in SOURCES]
    elif args.suite in ("mesh-only", "full-refinement"):
        setup = replace(base, nx=32,
                        spatial_azimuths=16 if args.suite == "full-refinement" else 8)
        label = "refined" if args.suite == "full-refinement" else "mesh32"
        cases = [(label, name, setup, profiles) for name in SOURCES]
    elif args.suite == "full-pulse":
        variants = {"mesh32": replace(base, nx=32),
                    "field-time": replace(base, field_cfl=.125),
                    "fluid-time": replace(base, hydro_cfl=.015),
                    "spatial": replace(base, radial_order=16, spatial_azimuths=16)}
        for label in args.variants or variants:
            cases += [(label, name, variants[label], profiles) for name in SOURCES]
    elif args.suite == "no-fields":
        setup = replace(base, electromagnetic=False)
        cases = [("no-fields", name, setup, profiles) for name in SOURCES]
    if args.variants and args.suite != "full-pulse":
        parser.error("--variants requires --suite full-pulse")
    cases = [case for case in cases if case[1] in args.sources]
    distributed = args.suite in ("mesh-only", "full-refinement", "full-pulse")
    launcher = shlex.split(args.launcher) if distributed else []
    (output/"metadata.json").write_text(json.dumps({
        "exe": str(exe), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "hostname": platform.node(), "suite": args.suite,
        "launcher": launcher,
        "profile_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in profiles.glob("*.dat")},
        "results_are_not_timing_samples": True,
        "cases": [{"label": label, "source": name, "setup": asdict(s)}
                  for label, name, s, _ in cases]}, indent=2)+"\n")
    with (output/"results.jsonl").open("w") as stream:
        for label, name, setup, source in cases:
            directory = output/f"{label}-{name}"
            deck = output/f"{label}-{name}.athinput"
            deck.write_text(input_text(name, source/f"{name}.dat", setup, True))
            result = run(exe, directory, ["-i", deck], args.timeout, launcher)
            result.update(analyze(directory, setup))
            result.update(label=label, name=name, setup=asdict(setup))
            result["input_sha256"] = hashlib.sha256(deck.read_bytes()).hexdigest()
            ledger = result["ledger"]
            expected = POWER_ERG_S*setup.power_scale*setup.duration_ps*1e-12
            if (not math.isclose(result["time"], setup.tlim_ps, rel_tol=2e-6) or
                not math.isclose(ledger["injected_erg"], expected, rel_tol=1e-10) or
                abs(ledger["residual_erg"])/expected > 1e-9):
                raise RuntimeError(f"Unfinished/nonconservative run: {directory}")
            if not math.isclose(result["mesh_deposited_fraction"],
                                ledger["deposited_erg"]/expected, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f"Mesh deposition/ledger mismatch: {directory}")
            if args.suite == "oracles":
                oracle = stopping_reference(np.loadtxt(source/f"{name}.dat"), setup)
                result["continuous_stopping_fraction"] = oracle
                result["stopping_error_absolute"] = abs(result["mesh_deposited_fraction"]-oracle)
                if result["stopping_error_absolute"] > .005:
                    raise RuntimeError(f"Stopping mismatch: {result}")
            (directory/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
            stream.write(json.dumps(result)+"\n")
            stream.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
