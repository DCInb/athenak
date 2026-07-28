"""MPI rank-0-only loading regression for two material opacity tables."""

import os
from pathlib import Path
import threading

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils
from test_suite.nr.mixed_thermal_radiation_utils import prepare_case


input_file = Path("mixed_thermal_radiation_mpi.athinput")
material0_table = Path("mixed_opacity_mpi_ch.dat")
material1_table = Path("mixed_opacity_mpi_he.dat")
material0_fifo = Path("mixed_opacity_ch.fifo")
material1_fifo = Path("mixed_opacity_he.fifo")


def run_case(basename, ranks, table0, table1):
    flags = [
        f"job/basename={basename}",
        "meshblock/nx1=8", "problem/yl=0.25", "problem/yr=0.25",
        f"materials/material0_opacity_table_file={table0}",
        f"materials/material1_opacity_table_file={table1}",
    ]
    assert testutils.mpi_run(
        str(input_file), flags=flags, threads=ranks, timeout=60.0), (
            f"{basename} failed with {ranks} MPI ranks.")
    return athena_read.tab(f"tab/{basename}.mhd_3t.00001.tab")


def compare_fields(candidate, reference):
    assert candidate.keys() == reference.keys()
    for field in reference:
        assert np.all(np.isfinite(candidate[field]))
        assert np.allclose(candidate[field], reference[field],
                           rtol=3.0e-12, atol=3.0e-13), field


def test_run():
    writers = []
    writer_errors = []
    try:
        prepare_case(input_file, material0_table, material1_table)
        reference = run_case(
            "mixed_opacity_mpi_1", 1, material0_table, material1_table)

        table_text = (
            material0_table.read_text(encoding="ascii"),
            material1_table.read_text(encoding="ascii"),
        )
        for fifo in (material0_fifo, material1_fifo):
            fifo.unlink(missing_ok=True)
            os.mkfifo(fifo)

        def write_once(path, content):
            try:
                with path.open("w", encoding="ascii") as stream:
                    stream.write(content)
            except Exception as exc:  # pragma: no cover - reported by main thread
                writer_errors.append(exc)

        for fifo, content in zip((material0_fifo, material1_fifo), table_text):
            writer = threading.Thread(
                target=write_once, args=(fifo, content), daemon=True)
            writer.start()
            writers.append(writer)

        parallel = run_case(
            "mixed_opacity_mpi_2", 2, material0_fifo, material1_fifo)
        for writer in writers:
            writer.join(timeout=10.0)
            assert not writer.is_alive(), "An opacity FIFO writer did not finish."
        assert not writer_errors, str(writer_errors[0]) if writer_errors else ""
        compare_fields(parallel, reference)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        for writer in writers:
            writer.join(timeout=1.0)
        for path in (input_file, material0_table, material1_table,
                     material0_fifo, material1_fifo):
            path.unlink(missing_ok=True)
        testutils.cleanup()
