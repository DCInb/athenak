"""MPI broadcast regression tests for tabulated thermal-radiation opacities."""

import os
from pathlib import Path
import threading

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/hydro/two_temperature_opacity_table.athinput"
opacity_table = "../../../inputs/hydro/two_temperature_opacity_table.dat"
fifo_table = Path("opacity_rank0.fifo")


def run_opacity_case(basename, ranks, filename):
    flags = [
        f"job/basename={basename}",
        f"thermal_radiation/opacity_table_file={filename}",
        "mesh/nx1=16",
        "meshblock/nx1=8",
    ]
    assert testutils.mpi_run(
        input_file, flags=flags, threads=ranks, timeout=60.0), (
            f"{basename} failed with {ranks} MPI ranks.")
    return athena_read.tab(f"tab/{basename}.hydro_3t.00001.tab")


def compare_fields(candidate, reference):
    assert candidate.keys() == reference.keys()
    for field in reference:
        assert np.all(np.isfinite(candidate[field]))
        assert np.allclose(candidate[field], reference[field],
                           rtol=2.0e-11, atol=2.0e-13), field


def test_run():
    writer = None
    writer_errors = []
    try:
        reference = run_opacity_case("opacity_mpi_1", 1, opacity_table)

        table_text = Path(opacity_table).read_text(encoding="ascii")
        fifo_table.unlink(missing_ok=True)
        os.mkfifo(fifo_table)

        def write_table_once():
            try:
                with fifo_table.open("w", encoding="ascii") as stream:
                    stream.write(table_text)
            except Exception as exc:  # pragma: no cover - reported in the main thread
                writer_errors.append(exc)

        writer = threading.Thread(target=write_table_once, daemon=True)
        writer.start()
        parallel = run_opacity_case("opacity_mpi_2", 2, fifo_table)
        writer.join(timeout=10.0)
        assert not writer.is_alive(), "FIFO table writer did not finish."
        assert not writer_errors, str(writer_errors[0]) if writer_errors else ""
        compare_fields(parallel, reference)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        if writer is not None:
            writer.join(timeout=1.0)
        fifo_table.unlink(missing_ok=True)
        testutils.cleanup()
