#!/usr/bin/env python3
"""Summarize Nsight SQLite exports, including CUDA-aware MPI staging.

Counts include initialization and shutdown. API, MPI, kernel, and copy durations
overlap; their sum is not an exclusive wall-time breakdown.
"""
import argparse
import json
from pathlib import Path
import re
import sqlite3


def summarize(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        tables = {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        kernels = {}
        for name, calls, seconds in connection.execute("""
            SELECT s.value, count(*), sum(k.end-k.start)*1e-9
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
            GROUP BY s.value
        """):
            match = re.search(r"fast_electrons::FastElectrons::(\w+)", name)
            key = match[1] if match else "Other kernels"
            entry = kernels.setdefault(key, {"calls": 0, "seconds": 0.0})
            entry["calls"] += calls
            entry["seconds"] += seconds
        copies = [dict(zip(("kind", "calls", "bytes", "seconds"), row))
                  for row in connection.execute("""
            SELECT e.label, count(*), sum(m.bytes), sum(m.end-m.start)*1e-9
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id GROUP BY e.label
        """)]
        apis = [dict(zip(("name", "calls", "seconds"), row))
                for row in connection.execute("""
            SELECT s.value, count(*), sum(r.end-r.start)*1e-9 AS seconds
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
            GROUP BY s.value ORDER BY seconds DESC
        """)]
        mpi = []
        for table in ("MPI_P2P_EVENTS", "MPI_COLLECTIVES_EVENTS", "MPI_START_WAIT_EVENTS"):
            if table in tables:
                mpi.extend(dict(zip(("name", "calls", "seconds"), row))
                           for row in connection.execute(f"""
                    SELECT s.value, count(*), sum(r.end-r.start)*1e-9
                    FROM {table} r JOIN StringIds s ON r.textId=s.id GROUP BY s.value
                """))
        devices = [r[0] for r in connection.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")]
    return {"trace": str(path), "kernels": kernels, "copies": copies, "apis": apis,
            "mpi": mpi, "devices": devices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, metavar="LABEL=SQLITE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for specification in args.profile:
        label, path = specification.split("=", 1)
        result[label] = summarize(Path(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
