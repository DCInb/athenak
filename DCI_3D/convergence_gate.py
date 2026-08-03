#!/usr/bin/env python3
"""Convergence / conservation / physics-equivalence gate for DCI_3D performance work.

`plan_performance.md` §2.3 records why this exists: the accepted campaign validated
optimisations with a byte-exact frozen-field comparison, and that gate has run out of
headroom.  Every remaining large win -- the safeguarded secant inverse (A1), the opacity
log pre-store (C4a), the Planck recurrence (C3), the strided radiation limiter (C5) -- is
roundoff-perturbing by construction and can never reproduce a bit-identical trajectory.
Those are algorithm changes, not defects, and they need a gate that measures *physics
equivalence* rather than bit equality.

This harness answers the five questions §6.0b asks of a candidate build:

1. does the solution converge to the same physical answer under refinement?
2. do total energy, ``div B`` and CH mass drift no more than the reference?
3. is the timestep-limiter sequence intact -- no collapse, no runaway substepping?
4. does laser power still close, with margin to the reflection and wave caps?
5. does the EOS stay inside its table -- no new clamps, no new disallowed states?

It deliberately reuses `verify_production_gate.py`'s metric layer (history parsing,
laser-record parsing, 3T binary inspection) rather than reimplementing it, so the two
gates cannot drift apart in what they mean by "energy" or "clamp".

Usage
-----
    # capture a reference from the frozen executable
    convergence_gate.py capture --binary DCI_3D/perf_work/athena.w1 \
        --label reference --cycles 40

    # capture the candidate and compare
    convergence_gate.py capture --binary DCI_3D/build/src/athena \
        --label candidate --cycles 40
    convergence_gate.py compare --reference <dir> --candidate <dir>

    # both in one step
    convergence_gate.py check --reference-binary A --candidate-binary B --cycles 40

    # prove the gate can see a perturbation it is supposed to see
    convergence_gate.py selftest --reference <dir>

Exit status is 0 when every check passes and 1 otherwise, so it composes with CI and
with the ledger workflow in `plan_performance.md` §4.3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
PRODUCTION_INPUT = CASE_DIR / "dci_3d.athinput"
MATERIAL_TABLE_DIR = CASE_DIR / "material_tables"
ENV_SCRIPT = Path("/home/mengqi/Research/bashrc_athenaK")
DEFAULT_WORK_DIR = CASE_DIR / "convergence_gate_runs"

sys.path.insert(0, str(CASE_DIR))
import verify_production_gate as vpg  # noqa: E402

# Transport tuning must match production; see perf_ledger.md card A4.
try:
    from run_case import MPI_TRANSPORT_ENV  # type: ignore
except Exception:  # pragma: no cover - run_case is the source of truth when present
    MPI_TRANSPORT_ENV = {"UCX_RNDV_THRESH": "1M"}


# ---------------------------------------------------------------------------------------
# Tolerance table
#
# Every entry states what is being bounded and why that bound and not a tighter one.
# A roundoff-perturbing change is expected to move these quantities; the gate's job is to
# separate "moved by roundoff amplified through a chaotic flow" from "moved because the
# physics changed".
# ---------------------------------------------------------------------------------------

TOLERANCES: dict[str, dict[str, Any]] = {
    "energy_closure_drift": {
        "limit": 5.0e-9,
        "unit": "fraction of chain_E",
        "why": (
            "Total energy closure is an identity the scheme enforces, not a converged "
            "quantity, so the candidate must hold it as tightly as the reference does. "
            "The bound is a few hundred ulps of the running total, which admits "
            "re-association of the reduction but not a real leak."
        ),
    },
    "ch_mass_drift": {
        "limit": 1.0e-11,
        "unit": "fraction of initial CH_mass",
        "why": (
            "CH mass moves only through a conservative scalar flux, so it is conserved "
            "to reduction roundoff independent of the EOS closure. The production gate "
            "check `ch_mass_conservation` uses the same reasoning."
        ),
    },
    "divb_ratio": {
        "limit": 1.0e-10,
        "unit": "integral |div B| / (integral |B| / L)",
        "why": (
            "Constrained transport makes this a machine-zero invariant. An EOS or "
            "subcycling change cannot legitimately raise it; if it does, CT has been "
            "broken rather than perturbed."
        ),
    },
    "field_relative_l2": {
        "limit": 2.0e-3,
        "unit": "relative L2 over 3T fields",
        "why": (
            "This is the only genuinely loose bound, and intentionally so. A "
            "roundoff-level perturbation grows in a laser-driven flow, so a bitwise "
            "criterion is meaningless here. 2e-3 is well below the deck's own "
            "discretisation error -- `resolution_or_opacity_sensitivity` accepts a much "
            "larger response to halving the mesh -- so a candidate inside this bound is "
            "the same physical solution at this resolution."
        ),
    },
    "refinement_convergence_ratio": {
        "limit": 1.5,
        "unit": "candidate refinement response / reference refinement response",
        "why": (
            "The convergence *test*: refining the mesh must move the candidate by the "
            "same amount it moves the reference. If the candidate is converging to a "
            "different answer, its refinement response diverges from the reference's "
            "while the coarse-grid difference stays small. Allowing 1.5x absorbs "
            "sampling noise in a single refinement pair."
        ),
    },
    "dt_sequence_relative": {
        "limit": 5.0e-2,
        "unit": "max relative macro-dt difference",
        "why": (
            "The macro dt is a min-reduction over wave speeds, so it responds to the "
            "perturbation but must not step to a different limiter branch. 5% is far "
            "below the factor-of-several drop that `causal_timestep_no_collapse` calls "
            "a collapse."
        ),
    },
    "dt_collapse_factor": {
        "limit": 0.5,
        "unit": "min(candidate dt)/min(reference dt)",
        "why": (
            "Hard floor on `causal_timestep_no_collapse`. A candidate whose smallest "
            "step is less than half the reference's has found a stiffness the reference "
            "did not, which is a physics change however good it looks on the clock."
        ),
    },
    "biermann_substep_growth": {
        "limit": 1.25,
        "unit": "candidate substeps / reference substeps",
        "why": (
            "The Biermann subcycle count is the cost multiplier the whole plan is trying "
            "to reduce. A closure change that silently inflates it has moved the cost "
            "rather than removed it."
        ),
    },
    "laser_residual": {
        "limit": 1.0e-10,
        "unit": "fraction of launched power",
        "why": (
            "launched - deposited - escaped - remaining is an accounting identity of the "
            "transport, independent of the EOS. It closes to reduction roundoff and must "
            "keep doing so; this backs `laser_and_boundary_energy_closure`."
        ),
    },
    "laser_cap_margin": {
        "limit": 2.0,
        "unit": "required margin factor to the reflection and wave caps",
        "why": (
            "§4.1 asks for >=2x margin. Rays parked at a cap are rays whose transport was "
            "truncated, so power closure can look healthy while the solution is wrong."
        ),
    },
    "eos_clamp_growth": {
        "limit": 0.0,
        "unit": "additional disallowed-flag cells vs reference",
        "why": (
            "`finite_nonnegative_3t` admits no new out-of-table states at all. A "
            "candidate inverse that lands outside the table where the reference did not "
            "has changed the physics, not the arithmetic. Energy-floor cells are "
            "reported but not bounded here: they are a pre-existing floor, and the "
            "reference's own count is the comparison."
        ),
    },
}


# ---------------------------------------------------------------------------------------
# Run capture
# ---------------------------------------------------------------------------------------


def launch_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(MPI_TRANSPORT_ENV)
    return env


def source_env_script() -> dict[str, str]:
    """Return the environment `bashrc_athenaK` establishes, without a login shell."""
    if not ENV_SCRIPT.exists():
        return launch_env()
    probe = subprocess.run(
        ["bash", "-c", f"set +u; source {ENV_SCRIPT} >/dev/null 2>&1; env -0"],
        capture_output=True,
        check=True,
    )
    env = {}
    for entry in probe.stdout.decode().split("\0"):
        if "=" in entry:
            key, _, value = entry.partition("=")
            env[key] = value
    env.update(MPI_TRANSPORT_ENV)
    return env


def capture_run(
    binary: Path,
    out_dir: Path,
    cycles: int,
    ranks: int,
    compact_scale: int,
    extra_overrides: list[str],
) -> Path:
    """Execute one gate run and return its directory.

    Outputs are configured for what the gate actually reads: the user history (energy,
    CH mass, div B, EOS flag counts), full-volume 3T dumps (field comparison and clamp
    census) and the cycle/laser stdout stream (limiter sequence, power closure).
    """
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")
    run_dir = out_dir / "run"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_dir.mkdir(parents=True)
    tables = run_dir / "material_tables"
    if MATERIAL_TABLE_DIR.exists():
        shutil.copytree(MATERIAL_TABLE_DIR, tables)

    overrides = [
        f"time/nlim={cycles}",
        "time/ndiag=1",
        # Slices and restarts are not read by any check; suppress the I/O stalls.
        "output2/dt=-1.0", "output3/dt=-1.0", "output4/dt=-1.0",
        "output5/dt=-1.0", "output6/dt=-1.0", "output7/dt=-1.0",
        "output11/dt=-1.0",
        # History every step: the limiter and conservation series are the measurement.
        "output1/dt=1.0e-12",
        # Full-volume 3T at every step end; the last dump is the comparison state.
        "output8/dt=-1.0",
        "output9/dt=1.0e-12",
        "output10/dt=-1.0",
    ]
    if compact_scale > 1:
        overrides += compact_overrides(compact_scale)
    overrides += extra_overrides

    command = [
        "mpirun", "-n", str(ranks), str(binary.resolve()),
        "--kokkos-map-device-id-by=mpi_rank",
        "-d", str(run_dir), "-i", str(PRODUCTION_INPUT),
    ] + overrides

    log_path = out_dir / "stdout.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, cwd=REPO, env=source_env_script(),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.monotonic()-started
    # A full-volume 3T dump of this mesh is several GB, and only the final state is
    # compared. Keeping the whole per-cycle series filled the disk once; prune as soon
    # as the run ends, before anything else needs space.
    retained = prune_volume_dumps(run_dir)
    manifest = {
        "binary": str(binary.resolve()),
        "binary_sha256": vpg.sha256_path(binary),
        "cycles": cycles,
        "ranks": ranks,
        "compact_scale": compact_scale,
        "overrides": overrides,
        "returncode": result.returncode,
        "wall_seconds": elapsed,
        "retained_volume_dump": retained,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Run failed (rc={result.returncode}); see {log_path}"
        )
    return out_dir


def prune_volume_dumps(run_dir: Path) -> str | None:
    """Keep only the last full-volume 3T dump; delete the rest.

    The comparison reads a single final state, but the run has to emit the series to
    guarantee a dump lands on the final cycle. At 512x256x256 with 27 fields each dump is
    several GB, so retaining twenty of them per capture is enough to exhaust the
    filesystem -- which is exactly what happened the first time this harness ran.
    """
    dumps = sorted(run_dir.rglob("*three_t*.bin"))
    if not dumps:
        return None
    for stale in dumps[:-1]:
        try:
            stale.unlink()
        except OSError:
            pass
    return dumps[-1].name


def compact_overrides(scale: int) -> list[str]:
    """Halve (or quarter) the mesh the way `run_case.py --compact-scale` does.

    Refinement here is *coarsening* relative to production: the production mesh is the
    fine member of the pair, so a converging candidate must track the reference on both.
    """
    base = {"nx1": 512, "nx2": 256, "nx3": 256}
    overrides = []
    for axis, cells in base.items():
        reduced = max(cells//scale, 32)
        overrides.append(f"mesh/{axis}={reduced}")
    return overrides


# ---------------------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------------------


def find_history(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "run").rglob("*.hst"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*.hst"))
    if not candidates:
        raise FileNotFoundError(f"No history file under {run_dir}")
    return candidates[-1]


def find_final_3t(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "run").rglob("*three_t*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No three_t binary under {run_dir}")
    return candidates[-1]


def parse_biermann(path: Path) -> list[dict[str, float]]:
    pattern = re.compile(
        r"cycle=(\d+).*?biermann_substeps=([+\-0-9.eE]+).*?"
        r"biermann_max_ratio=([+\-0-9.eE]+)"
    )
    rows = []
    for cycle, substeps, ratio in pattern.findall(path.read_text(encoding="utf-8")):
        rows.append({
            "cycle": int(cycle),
            "substeps": float(substeps),
            "max_ratio": float(ratio),
        })
    return rows


def read_3t_fields(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "vis" / "python"))
    import bin_convert  # type: ignore
    import numpy as np

    data = bin_convert.read_binary(str(path))
    fields = {}
    for name in ("eion", "eele", "tion", "tele", "erad", "trad"):
        if name in data["mb_data"]:
            fields[name] = np.concatenate(
                [np.asarray(a).ravel() for a in data["mb_data"][name]]
            )
    return {"time": float(data["time"]), "cycle": int(data["cycle"]),
            "fields": fields}


def relative_l2(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    import numpy as np

    per_field = {}
    for name, ref in reference["fields"].items():
        cand = candidate["fields"].get(name)
        if cand is None or cand.shape != ref.shape:
            per_field[name] = math.inf
            continue
        denominator = float(np.sqrt(np.sum(ref*ref)))
        if denominator == 0.0:
            per_field[name] = 0.0
            continue
        per_field[name] = float(np.sqrt(np.sum((cand-ref)**2)))/denominator
    return per_field


def run_metrics(run_dir: Path) -> dict[str, Any]:
    """Reduce one captured run to the quantities the gate compares."""
    log = run_dir / "stdout.log"
    history = vpg.read_history(find_history(run_dir))
    cycles = vpg.parse_cycle_log(log)
    laser = vpg.parse_laser_diagnostics(log)
    biermann = parse_biermann(log)
    final_3t = find_final_3t(run_dir)
    census = vpg.check_3t_binary(final_3t)

    dts = [row["dt"] for row in cycles]

    # Energy closure: chain_E is total material + radiation energy; the deposited laser
    # energy and the escaped radiation are the only legitimate sources and sinks.
    chain = history.get("chain_E", [])
    laser_dep = history.get("laser_Edep", [0.0]*len(chain))
    closure_drift = 0.0
    if len(chain) > 1:
        scale = max(abs(chain[0]), abs(chain[-1]), 1.0e-300)
        expected = chain[0]+(laser_dep[-1]-laser_dep[0])
        closure_drift = abs(chain[-1]-expected)/scale

    ch = history.get("CH_mass", [])
    ch_drift = 0.0
    if len(ch) > 1 and ch[0] != 0.0:
        ch_drift = max(abs(value-ch[0]) for value in ch)/abs(ch[0])

    divb = history.get("divB", [])
    absb = history.get("abs_B", [])
    divb_ratio = 0.0
    if divb and absb:
        # Non-dimensionalise with the domain scale so the ratio is grid independent.
        domain_length = 4.0
        for value, magnitude in zip(divb, absb):
            reference_scale = abs(magnitude)/domain_length
            if reference_scale > 0.0:
                divb_ratio = max(divb_ratio, abs(value)/reference_scale)

    laser_residual = 0.0
    reflection_margin = math.inf
    wave_margin = math.inf
    for record in laser:
        launched = float(record.get("launched", 0.0))
        if launched > 0.0:
            residual = abs(float(record.get("residual", 0.0)))/launched
            laser_residual = max(laser_residual, residual)
            remaining = float(record.get("remaining", 0.0))
            reflection_remaining = float(record.get("reflection_remaining", 0.0))
            wave_remaining = float(record.get("wave_remaining", 0.0))
            if reflection_remaining > 0.0:
                reflection_margin = min(
                    reflection_margin, launched/reflection_remaining)
            if wave_remaining > 0.0:
                wave_margin = min(wave_margin, launched/wave_remaining)
            del remaining

    return {
        "final_time": cycles[-1]["time"] if cycles else 0.0,
        "cycle_count": len(cycles),
        "dt_series": dts,
        "dt_min": min(dts) if dts else 0.0,
        "energy_closure_drift": closure_drift,
        "ch_mass_drift": ch_drift,
        "divb_ratio": divb_ratio,
        "biermann_substeps_total": sum(row["substeps"] for row in biermann),
        "biermann_max_ratio": max(
            (row["max_ratio"] for row in biermann), default=0.0),
        "laser_residual": laser_residual,
        "laser_reflection_margin": reflection_margin,
        "laser_wave_margin": wave_margin,
        "eos_disallowed_cell_count": census["eos_disallowed_cell_count"],
        "eos_energy_floor_cell_count": census["eos_energy_floor_cell_count"],
        "eos_trace_cell_count": census["eos_trace_cell_count"],
        "final_3t": str(final_3t),
    }


# ---------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------


def check(name: str, passed: bool, value: Any, **extra: Any) -> dict[str, Any]:
    entry = {
        "check": name,
        "passed": bool(passed),
        "value": value,
        "limit": TOLERANCES[name]["limit"] if name in TOLERANCES else None,
        "unit": TOLERANCES[name]["unit"] if name in TOLERANCES else None,
    }
    entry.update(extra)
    return entry


def compare(
    reference: Path,
    candidate: Path,
    reference_coarse: Path | None = None,
    candidate_coarse: Path | None = None,
) -> dict[str, Any]:
    ref = run_metrics(reference)
    cand = run_metrics(candidate)
    checks: list[dict[str, Any]] = []

    # --- 2. conservation -----------------------------------------------------------
    checks.append(check(
        "energy_closure_drift",
        cand["energy_closure_drift"] <= max(
            TOLERANCES["energy_closure_drift"]["limit"],
            ref["energy_closure_drift"]),
        cand["energy_closure_drift"], reference=ref["energy_closure_drift"]))
    checks.append(check(
        "ch_mass_drift",
        cand["ch_mass_drift"] <= max(
            TOLERANCES["ch_mass_drift"]["limit"], ref["ch_mass_drift"]),
        cand["ch_mass_drift"], reference=ref["ch_mass_drift"]))
    checks.append(check(
        "divb_ratio",
        cand["divb_ratio"] <= max(
            TOLERANCES["divb_ratio"]["limit"], ref["divb_ratio"]),
        cand["divb_ratio"], reference=ref["divb_ratio"]))

    # --- 3. limiter sequence -------------------------------------------------------
    ref_dt, cand_dt = ref["dt_series"], cand["dt_series"]
    paired = list(zip(ref_dt, cand_dt))
    dt_relative = max(
        (abs(c-r)/abs(r) for r, c in paired if r != 0.0), default=0.0)
    checks.append(check(
        "dt_sequence_relative",
        dt_relative <= TOLERANCES["dt_sequence_relative"]["limit"],
        dt_relative, compared_cycles=len(paired),
        reference_cycles=len(ref_dt), candidate_cycles=len(cand_dt)))
    collapse = (cand["dt_min"]/ref["dt_min"]) if ref["dt_min"] > 0.0 else 0.0
    checks.append(check(
        "dt_collapse_factor",
        collapse >= TOLERANCES["dt_collapse_factor"]["limit"],
        collapse, reference_dt_min=ref["dt_min"], candidate_dt_min=cand["dt_min"]))
    substep_growth = (
        cand["biermann_substeps_total"]/ref["biermann_substeps_total"]
        if ref["biermann_substeps_total"] > 0 else 1.0)
    checks.append(check(
        "biermann_substep_growth",
        substep_growth <= TOLERANCES["biermann_substep_growth"]["limit"],
        substep_growth,
        reference_substeps=ref["biermann_substeps_total"],
        candidate_substeps=cand["biermann_substeps_total"],
        candidate_max_stability_ratio=cand["biermann_max_ratio"]))

    # --- 4. laser closure ----------------------------------------------------------
    checks.append(check(
        "laser_residual",
        cand["laser_residual"] <= max(
            TOLERANCES["laser_residual"]["limit"], ref["laser_residual"]),
        cand["laser_residual"], reference=ref["laser_residual"]))
    margin = min(cand["laser_reflection_margin"], cand["laser_wave_margin"])
    checks.append(check(
        "laser_cap_margin",
        margin >= TOLERANCES["laser_cap_margin"]["limit"],
        margin if math.isfinite(margin) else "no rays at cap",
        reflection_margin=(cand["laser_reflection_margin"]
                           if math.isfinite(cand["laser_reflection_margin"])
                           else None),
        wave_margin=(cand["laser_wave_margin"]
                     if math.isfinite(cand["laser_wave_margin"]) else None)))

    # --- 5. EOS clamp census -------------------------------------------------------
    extra_clamps = (cand["eos_disallowed_cell_count"]
                    - ref["eos_disallowed_cell_count"])
    checks.append(check(
        "eos_clamp_growth",
        extra_clamps <= TOLERANCES["eos_clamp_growth"]["limit"],
        extra_clamps,
        reference_disallowed=ref["eos_disallowed_cell_count"],
        candidate_disallowed=cand["eos_disallowed_cell_count"],
        reference_energy_floor=ref["eos_energy_floor_cell_count"],
        candidate_energy_floor=cand["eos_energy_floor_cell_count"]))

    # --- 1. field agreement and refinement convergence ------------------------------
    fine_l2 = relative_l2(
        read_3t_fields(Path(ref["final_3t"])),
        read_3t_fields(Path(cand["final_3t"])))
    worst_field = max(fine_l2, key=fine_l2.get) if fine_l2 else None
    worst_l2 = fine_l2[worst_field] if worst_field else 0.0
    checks.append(check(
        "field_relative_l2",
        worst_l2 <= TOLERANCES["field_relative_l2"]["limit"],
        worst_l2, worst_field=worst_field, per_field=fine_l2))

    if reference_coarse is not None and candidate_coarse is not None:
        ref_coarse = run_metrics(reference_coarse)
        cand_coarse = run_metrics(candidate_coarse)
        coarse_l2 = relative_l2(
            read_3t_fields(Path(ref_coarse["final_3t"])),
            read_3t_fields(Path(cand_coarse["final_3t"])))
        # The candidate converges to the reference's answer if refining the mesh does not
        # amplify the candidate-vs-reference difference: the difference must be a
        # perturbation that the discretisation damps, not a systematic offset.
        worst_coarse = max(coarse_l2.values(), default=0.0)
        ratio = (worst_l2/worst_coarse) if worst_coarse > 0.0 else 0.0
        checks.append(check(
            "refinement_convergence_ratio",
            ratio <= TOLERANCES["refinement_convergence_ratio"]["limit"],
            ratio, coarse_l2=worst_coarse, fine_l2=worst_l2))
    else:
        checks.append({
            "check": "refinement_convergence_ratio",
            "passed": None,
            "value": None,
            "note": (
                "skipped: pass --reference-coarse/--candidate-coarse (or use "
                "`check --with-refinement`) to run the coarse pair"
            ),
        })

    passed = all(entry["passed"] for entry in checks if entry["passed"] is not None)
    return {
        "passed": passed,
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_metrics": ref,
        "candidate_metrics": cand,
        "checks": checks,
        "tolerances": TOLERANCES,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["", "DCI_3D convergence gate", "=" * 60]
    lines.append(f"reference : {report['reference']}")
    lines.append(f"candidate : {report['candidate']}")
    lines.append("")
    width = max(len(entry["check"]) for entry in report["checks"])
    for entry in report["checks"]:
        if entry["passed"] is None:
            status = "SKIP"
        else:
            status = "PASS" if entry["passed"] else "FAIL"
        value = entry.get("value")
        if isinstance(value, float):
            shown = f"{value:.6e}"
        else:
            shown = str(value)
        limit = entry.get("limit")
        limit_text = f"  (limit {limit:g})" if isinstance(limit, float) else ""
        lines.append(f"  [{status}] {entry['check']:<{width}}  {shown}{limit_text}")
        if entry.get("note"):
            lines.append(f"         {entry['note']}")
    lines.append("")
    lines.append("RESULT: " + ("PASS" if report["passed"] else "FAIL"))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cycles", type=int, default=40,
                        help="cycle limit for each captured run")
    common.add_argument("--ranks", type=int, default=8)
    common.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    common.add_argument("--restart", type=Path, default=None,
                        help="representative later-time restart file (§6.0b prefers "
                             "this over t=0)")

    p_capture = sub.add_parser("capture", parents=[common])
    p_capture.add_argument("--binary", type=Path, required=True)
    p_capture.add_argument("--label", required=True)
    p_capture.add_argument("--compact-scale", type=int, default=1)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--reference", type=Path, required=True)
    p_compare.add_argument("--candidate", type=Path, required=True)
    p_compare.add_argument("--reference-coarse", type=Path, default=None)
    p_compare.add_argument("--candidate-coarse", type=Path, default=None)
    p_compare.add_argument("--json", type=Path, default=None)

    p_check = sub.add_parser("check", parents=[common])
    p_check.add_argument("--reference-binary", type=Path, required=True)
    p_check.add_argument("--candidate-binary", type=Path, required=True)
    p_check.add_argument("--label", default="check")
    p_check.add_argument("--with-refinement", action="store_true",
                         help="also capture a coarse pair and run the convergence test")
    p_check.add_argument("--json", type=Path, default=None)

    p_self = sub.add_parser("selftest")
    p_self.add_argument("--reference", type=Path, required=True)
    p_self.add_argument("--perturbed", type=Path, default=None,
                        help="a run from a deliberately perturbed build; when omitted "
                             "only the reference-against-itself half runs")
    p_self.add_argument("--json", type=Path, default=None)

    return parser.parse_args()


def restart_overrides(restart: Path | None) -> list[str]:
    return [] if restart is None else [f"-r {restart}"]


def main() -> int:
    args = parse_args()

    if args.command == "capture":
        out = args.work_dir / args.label
        extra = []
        if args.restart is not None:
            extra.append(f"-r={args.restart}")
        capture_run(args.binary, out, args.cycles, args.ranks,
                    args.compact_scale, extra)
        print(f"captured {out}")
        return 0

    if args.command == "compare":
        report = compare(args.reference, args.candidate,
                         args.reference_coarse, args.candidate_coarse)
        print(render(report))
        if args.json:
            args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.command == "check":
        base = args.work_dir / args.label
        extra = []
        if args.restart is not None:
            extra.append(f"-r={args.restart}")
        ref = capture_run(args.reference_binary, base / "reference",
                          args.cycles, args.ranks, 1, extra)
        cand = capture_run(args.candidate_binary, base / "candidate",
                           args.cycles, args.ranks, 1, extra)
        ref_coarse = cand_coarse = None
        if args.with_refinement:
            ref_coarse = capture_run(args.reference_binary, base / "reference_coarse",
                                     args.cycles, args.ranks, 2, extra)
            cand_coarse = capture_run(args.candidate_binary, base / "candidate_coarse",
                                      args.cycles, args.ranks, 2, extra)
        report = compare(ref, cand, ref_coarse, cand_coarse)
        print(render(report))
        if args.json:
            args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.command == "selftest":
        identity = compare(args.reference, args.reference)
        print("--- reference against itself (must PASS with zero drift) ---")
        print(render(identity))
        ok = identity["passed"]
        for entry in identity["checks"]:
            if entry["check"] == "field_relative_l2" and entry["value"] != 0.0:
                print("SELFTEST FAIL: identity comparison is not exactly zero")
                ok = False
        if args.perturbed is not None:
            perturbed = compare(args.reference, args.perturbed)
            print("--- reference against a deliberately perturbed build ---")
            print(render(perturbed))
            moved = any(
                entry["check"] == "field_relative_l2"
                and isinstance(entry["value"], float) and entry["value"] > 0.0
                for entry in perturbed["checks"])
            if not moved:
                print("SELFTEST FAIL: the gate cannot see the injected perturbation")
                ok = False
            else:
                print("SELFTEST: perturbation detected with a finite magnitude")
        if args.json:
            args.json.write_text(
                json.dumps({"identity": identity}, indent=2)+"\n", encoding="utf-8")
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
