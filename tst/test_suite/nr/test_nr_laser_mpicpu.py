"""MPI migration regression tests for the two-temperature laser module."""

import shutil

import numpy as np
import pytest

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_cpu import (
    assemble_binary_field,
    read_laser_binary,
)


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"
pulse_file = "../../../tst/inputs/laser_multiknot.pulse"


def run_mpi_case(basename, ranks, flags):
    common = [
        f"job/basename={basename}",
        "time/integrator=rk1",
        "time/tlim=1.0e-7",
        "output1/dt=-1.0",
        "output2/dt=-1.0",
        "output3/dt=-1.0",
        "output4/dt=1.0e-7",
        "laser/report_diagnostics=true",
    ]
    assert testutils.mpi_run(
        input_file, flags=common + flags, threads=ranks), (
            f"{basename} failed with {ranks} MPI ranks.")
    return read_laser_binary(f"bin/{basename}.laser_full.00001.bin")


def fields(output):
    return {
        name: assemble_binary_field(output, name)
        for name in ("laser_q", "laser_ray_count", "laser_tau", "laser_path")
    }


def compare_fields(candidate, reference):
    for name in reference:
        assert np.allclose(candidate[name], reference[name],
                           rtol=2.0e-11, atol=2.0e-13), name


def test_run():
    try:
        # Eight x1 MeshBlocks force one or more migration waves for 2, 4, and 8 ranks.
        slab_flags = [
            "mesh/nx1=64", "mesh/nx2=1", "mesh/nx3=1",
            "meshblock/nx1=8", "meshblock/nx2=1", "meshblock/nx3=1",
            "laser/beam0_nrays=13",
            "laser/absorption_coefficient=2.0",
            "laser/max_segments_per_launch=3",
            "laser/max_transport_iterations=8",
            "laser/gpu_aware_mpi=false",
        ]
        reference = fields(run_mpi_case("laser_mpi_slab_1", 1, slab_flags))
        for ranks in (2, 4, 8):
            candidate = fields(run_mpi_case(
                f"laser_mpi_slab_{ranks}", ranks, slab_flags))
            compare_fields(candidate, reference)

        # A nearly boundary-parallel beam crosses many rank regions in two dimensions.
        stress_flags = [
            "mesh/nx1=64", "mesh/nx2=64", "mesh/nx3=1",
            "meshblock/nx1=16", "meshblock/nx2=16", "meshblock/nx3=1",
            "laser/beam0_nrays=1",
            "laser/beam0_origin_x2=-0.49",
            "laser/beam0_direction_x1=0.03",
            "laser/beam0_direction_x2=1.0",
            "laser/absorption_coefficient=0.25",
            "laser/max_segments_per_launch=2",
            "laser/max_transport_iterations=16",
            "laser/gpu_aware_mpi=false",
        ]
        stress_reference = fields(run_mpi_case(
            "laser_mpi_stress_1", 1, stress_flags))
        stress_parallel = fields(run_mpi_case(
            "laser_mpi_stress_4", 4, stress_flags))
        compare_fields(stress_parallel, stress_reference)

        # On a CPU build, device memory is host-accessible, so this also exercises the
        # direct-buffer branch used by GPU-aware MPI implementations.
        direct_flags = slab_flags + ["laser/gpu_aware_mpi=true"]
        direct = fields(run_mpi_case("laser_mpi_direct_2", 2, direct_flags))
        compare_fields(direct, reference)

        # A rank-0 pulse-file read and an external focused lens are invariant to
        # the two-rank decomposition.
        lens_flags = [
            "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=1",
            "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=1",
            "laser/beam0_nrays=37",
            "laser/beam0_geometry=lens",
            "laser/beam0_lens_x1=-0.25",
            "laser/beam0_lens_x2=0.0",
            "laser/beam0_target_x1=0.75",
            "laser/beam0_target_x2=0.0",
            "laser/beam0_radius=0.25",
            "laser/beam0_target_radius=0.05",
            "laser/beam0_profile=gaussian",
            "laser/beam0_profile_radius=0.12",
            f"laser/beam0_pulse_file={pulse_file}",
            "laser/beam0_pulse_mode=relative",
            "laser/absorption_coefficient=0.5",
            "laser/max_segments_per_launch=4",
            "laser/max_transport_iterations=16",
            "laser/gpu_aware_mpi=false",
        ]
        lens_reference = fields(run_mpi_case(
            "laser_mpi_lens_pulse_1", 1, lens_flags))
        lens_parallel = fields(run_mpi_case(
            "laser_mpi_lens_pulse_2", 2, lens_flags))
        compare_fields(lens_parallel, lens_reference)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
        shutil.rmtree("bin", ignore_errors=True)
