"""MPI migration regression tests for the two-temperature laser module."""

import os
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


def run_mpi_case(basename, ranks, flags, dump=1):
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
    return read_laser_binary(f"bin/{basename}.laser_full.{dump:05d}.bin")


def fields(output):
    return {
        name: assemble_binary_field(output, name)
        for name in (
            "laser_q", "laser_energy", "laser_ray_count", "laser_tau", "laser_path"
        )
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

        # Global ray ownership must use the exact same face coordinates as MeshBlock.
        # On this asymmetric ten-block, four-rank domain, a nearly face-tangent ray
        # crosses the rank-2/rank-1 face at x=-0.6. Its small negative x direction makes
        # the forward ownership probe about two ulps long: enough to cross MeshBlock's
        # canonical face but not an algebraically reconstructed, shifted global face.
        asymmetric_flags = [
            "mesh/nx1=80", "mesh/nx2=8", "mesh/nx3=1",
            "mesh/x1min=-2.0", "mesh/x1max=1.5",
            "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=1",
            "laser/beam0_nrays=1",
            "laser/beam0_origin_x1=-0.599",
            "laser/beam0_origin_x2=-0.25",
            "laser/beam0_direction_x1=-0.007654894771870187",
            "laser/beam0_direction_x2=1.0",
            "laser/absorption_coefficient=0.25",
            "laser/max_segments_per_launch=3",
            "laser/max_transport_iterations=8",
            "laser/gpu_aware_mpi=false",
        ]
        asymmetric_reference = fields(run_mpi_case(
            "laser_mpi_asymmetric_1", 1, asymmetric_flags))
        asymmetric_parallel = fields(run_mpi_case(
            "laser_mpi_asymmetric_4", 4, asymmetric_flags))
        compare_fields(asymmetric_parallel, asymmetric_reference)

        # A forward probe must move by at least one representable value in every
        # nonzero direction.  This ray reaches the x=0.5 block face from one ulp
        # above it; probe*direction is too small to change x in binary64.  A raw
        # coordinate addition therefore reselects the departing block and exhausts
        # the wave cap without another finite segment.
        subulp_probe_flags = [
            "mesh/nx1=16", "mesh/nx2=8", "mesh/nx3=1",
            "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=1",
            "laser/beam0_nrays=1",
            "laser/beam0_origin_x1=0.5000000000000001",
            "laser/beam0_origin_x2=0.0",
            "laser/beam0_direction_x1=-1.0e-8",
            "laser/beam0_direction_x2=1.0",
            "laser/absorption_coefficient=0.25",
            "laser/max_segments_per_launch=8",
            "laser/max_transport_iterations=1",
            "laser/max_mpi_waves=2",
            "laser/gpu_aware_mpi=false",
        ]
        subulp_reference = fields(run_mpi_case(
            "laser_mpi_subulp_probe_1", 1, subulp_probe_flags))
        subulp_parallel = fields(run_mpi_case(
            "laser_mpi_subulp_probe_2", 2, subulp_probe_flags))
        compare_fields(subulp_parallel, subulp_reference)

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

        # Both ranks take the same active-to-inactive branch. Only the two powered
        # RK2 stages run transport; the following dark stages preserve cumulative energy.
        offpulse_flags = [
            "time/integrator=rk2",
            "time/initial_dt=1.0e-6",
            "time/tlim=2.0e-6",
            "output4/dt=1.0e-6",
            "laser/beam0_nrays=13",
            "laser/absorption_coefficient=2.0",
            f"laser/beam0_pulse_file={pulse_file}",
            "laser/beam0_pulse_mode=relative",
        ]
        offpulse_results = []
        for ranks in (1, 2):
            log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
            result = fields(run_mpi_case(
                f"laser_mpi_offpulse_{ranks}", ranks, offpulse_flags, dump=2))
            with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
                stream.seek(log_offset)
                appended_log = stream.read()
            assert appended_log.count("laser: launched=") == 2
            assert np.count_nonzero(result["laser_q"]) == 0
            assert np.any(result["laser_energy"] > 0.0)
            offpulse_results.append(result)
        compare_fields(offpulse_results[1], offpulse_results[0])
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
        shutil.rmtree("bin", ignore_errors=True)
