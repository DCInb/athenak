#!/usr/bin/env python3
"""Run the FLASH quadratic laser-tube benchmark on AthenaK GPUs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import time


BENCH_DIR = Path(__file__).resolve().parent
TEST_DIR = BENCH_DIR.parent
REPO = TEST_DIR.parent
BINARY = TEST_DIR / "build" / "src" / "athena"
HELPER = Path("/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh")

CASES = (
    ("r32_gpu1", BENCH_DIR / "inputs" / "flash_laser_tube_r32.athinput", 1),
    ("r64_gpu1", BENCH_DIR / "inputs" / "flash_laser_tube_r64.athinput", 1),
    ("r64_gpu2", BENCH_DIR / "inputs" / "flash_laser_tube_r64.athinput", 2),
    ("r64_gpu4", BENCH_DIR / "inputs" / "flash_laser_tube_r64.athinput", 4),
)


def main() -> int:
    devices = [item.strip() for item in
               os.environ.get("ATHENAK_TEST_GPUS", "2,3,4,5").split(",")
               if item.strip()]
    if len(devices) < 4:
        raise RuntimeError("ATHENAK_TEST_GPUS must provide four GPU IDs")
    if not BINARY.is_file():
        raise RuntimeError(f"Missing AthenaK benchmark binary: {BINARY}")

    (BENCH_DIR / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, input_file, ranks in CASES:
        run_dir = BENCH_DIR / "runs" / name
        src_dir = run_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        link = src_dir / "athena"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(BINARY)
        generated = [item for item in run_dir.iterdir() if item.name != "src"]
        if generated:
            raise RuntimeError(f"Refusing to overwrite {run_dir}: {generated}")

        visible = devices[:ranks]
        log_path = BENCH_DIR / "logs" / f"{name}.log"
        command = [str(HELPER), "run", "--problem", "tests/laser",
                   "--repo", str(REPO), "--build-dir", str(run_dir),
                   "--input", str(input_file), "--ranks", str(ranks)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
        print(f"START case={name} visible={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        start = time.monotonic()
        with log_path.open("w") as log:
            log.write("command=" + " ".join(command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            log.flush()
            result = subprocess.run(command, cwd=REPO, env=env, text=True,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
        elapsed = time.monotonic() - start
        rows.append({"case": name, "resolution": 32 if name.startswith("r32") else 64,
                     "gpus": ranks, "visible_devices": env["CUDA_VISIBLE_DEVICES"],
                     "input": str(input_file), "run_dir": str(run_dir),
                     "log": str(log_path), "exit_code": result.returncode,
                     "elapsed_seconds": f"{elapsed:.6f}"})
        print(f"DONE case={name} exit={result.returncode} elapsed={elapsed:.3f}s",
              flush=True)

    with (BENCH_DIR / "run_status.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 1 if any(row["exit_code"] != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
