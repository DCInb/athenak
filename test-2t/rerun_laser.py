#!/usr/bin/env python3
"""Run only the isolated laser regression and replace its status-matrix rows."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import time

from run_all import BUILD_BINARY, FEATURES, GPU_COUNTS, HELPER, REPO, TEST_DIR


def main() -> int:
    feature, input_file, helper_problem = next(
        item for item in FEATURES if item[0] == "laser")
    devices = [entry.strip() for entry in
               os.environ.get("ATHENAK_TEST_GPUS", "2,3,4,5").split(",")
               if entry.strip()]
    new_rows = []
    for ranks in GPU_COUNTS:
        run_dir = TEST_DIR / "runs" / "laser_regression_complete" / f"gpu{ranks}"
        src_dir = run_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        link = src_dir / "athena"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(BUILD_BINARY)
        generated = [path for path in run_dir.iterdir() if path.name != "src"]
        if generated:
            raise RuntimeError(f"Refusing to overwrite {run_dir}: {generated}")
        visible = devices[:ranks]
        log_path = TEST_DIR / "logs" / f"laser_gpu{ranks}.log"
        command = [str(HELPER), "run", "--problem", helper_problem,
                   "--repo", str(REPO), "--build-dir", str(run_dir),
                   "--input", str(input_file), "--ranks", str(ranks)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
        start = time.monotonic()
        print(f"START feature=laser gpus={ranks} visible={env['CUDA_VISIBLE_DEVICES']}",
              flush=True)
        with log_path.open("w") as log:
            log.write("command=" + " ".join(command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            log.flush()
            result = subprocess.run(command, cwd=REPO, env=env, text=True,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
        elapsed = time.monotonic() - start
        new_rows.append({
            "feature": feature,
            "gpus": ranks,
            "visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "input": str(input_file),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "exit_code": result.returncode,
            "elapsed_seconds": f"{elapsed:.6f}",
        })
        print(f"DONE feature=laser gpus={ranks} exit={result.returncode} "
              f"elapsed={elapsed:.3f}s", flush=True)

    status_path = TEST_DIR / "run_status.csv"
    old_rows = list(csv.DictReader(status_path.open()))
    rows = [row for row in old_rows if row["feature"] != "laser"] + new_rows
    order = {name: index for index, (name, _, _) in enumerate(FEATURES)}
    rows.sort(key=lambda row: (order[row["feature"]], int(row["gpus"])))
    with status_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 1 if any(row["exit_code"] != 0 for row in new_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
