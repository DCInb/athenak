"""Static-refinement coverage for the complete Biermann subcycle."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
FINAL_TIME = 5.0e-3
BASENAMES = (
    "biermann_subcycle_smr_dual",
    "biermann_subcycle_smr_nodual",
    "biermann_subcycle_smr_dual_branch",
    "biermann_subcycle_smr_magnetization_floor",
)


def run_case(basename, dual_energy, extra_flags=None):
    flags = [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=8", "meshblock/nx2=8",
        "mesh_refinement/refinement=static",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        f"mhd/dual_energy={str(dual_energy).lower()}",
        "time/align_outputs=true",
        f"time/tlim={FINAL_TIME}",
        "output1/dt=0.001",
        f"output2/dt={FINAL_TIME}", "output2/data_format=%24.17e",
        f"output3/dt={FINAL_TIME}", "output3/data_format=%24.17e",
        f"output4/dt={FINAL_TIME}",
        f"output8/dt={FINAL_TIME}",
    ]
    if extra_flags:
        flags.extend(extra_flags)
    assert testutils.run(INPUT_FILE, flags=flags, timeout=120.0), (
        f"{basename} failed")

    field = athena_read.tab(f"tab/{basename}.biermann.00001.tab")
    two_temperature = athena_read.tab(
        f"tab/{basename}.two_temperature.00001.tab")
    history = athena_read.hst(f"{basename}.mhd.hst")
    divb_output = bin_convert.read_binary(f"bin/{basename}.divb.00001.bin")
    conserved = bin_convert.read_binary(
        f"bin/{basename}.biermann_conserved.00001.bin")
    return field, two_temperature, history, divb_output, conserved


def relative_l2(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right))
    assert scale > 0.0
    return np.linalg.norm(left-right)/scale


def assert_invariants(result, expected_cycles=5):
    field, two_temperature, history, divb_output, conserved = result
    assert len(history["time"])-1 == expected_cycles
    assert np.all(np.isfinite(history["tot-E"]))
    assert np.allclose(history["tot-E"], history["tot-E"][0],
                       rtol=3.0e-12, atol=3.0e-12)

    for key in ("dens", "eint"):
        assert np.all(np.isfinite(field[key]))
        assert np.all(field[key] > 0.0)
    for key in ("eion", "eele", "tion", "tele"):
        assert np.all(np.isfinite(two_temperature[key]))
        assert np.all(two_temperature[key] > 0.0)
    assert np.max(np.abs(field["bcc3"])) > 1.0e-3
    assert np.allclose(
        field["dens"]*(two_temperature["eion"]+two_temperature["eele"]),
        field["eint"], rtol=3.0e-11, atol=3.0e-12)

    levels = np.asarray(divb_output["mb_logical"])[:, 3]
    unique, counts = np.unique(levels, return_counts=True)
    assert set(unique.tolist()) == {0, 1}, (unique, counts)
    assert np.all(counts > 0), counts
    divb = np.concatenate([
        np.asarray(block).ravel()
        for block in divb_output["mb_data"]["divb"]
    ])
    assert np.all(np.isfinite(divb))
    assert np.max(np.abs(divb)) < 5.0e-13


def assert_dual_energy_branch_flip(result, eta1):
    """Require conservative and auxiliary C2P branches in one accepted AMR state."""
    conserved = result[4]

    def flattened(name):
        return np.concatenate([
            np.asarray(block).ravel()
            for block in conserved["mb_data"][name]
        ])

    density = flattened("dens")
    momentum_squared = sum(flattened(name)**2
                           for name in ("mom1", "mom2", "mom3"))
    magnetic_squared = sum(flattened(name)**2
                           for name in ("bcc1", "bcc2", "bcc3"))
    total_energy = flattened("ener")
    conservative_internal = (
        total_energy-0.5*momentum_squared/density-0.5*magnetic_squared)
    conservative_fraction = conservative_internal/np.maximum(
        total_energy, 1.0e-18)
    assert np.any((conservative_internal > 0.0) &
                  (conservative_fraction > eta1))
    assert np.any((conservative_internal <= 0.0) |
                  (conservative_fraction <= eta1))


def assert_magnetization_floor(result, sigma_max):
    conserved = result[4]

    def flattened(name):
        return np.concatenate([
            np.asarray(block).ravel()
            for block in conserved["mb_data"][name]
        ])

    density = flattened("dens")
    magnetic_squared = sum(flattened(name)**2
                           for name in ("bcc1", "bcc2", "bcc3"))
    active = magnetic_squared > 0.5
    assert np.any(active)
    required_density = magnetic_squared[active]/sigma_max
    assert np.all(density[active] >= required_density*(1.0-3.0e-12))
    # The tiny accepted step evolves B after the density floor is applied, so the
    # final ratio need only remain close to the initially saturated value.
    assert np.min(density[active]/required_density) < 1.0+1.0e-5


def test_run():
    try:
        results = [
            run_case(BASENAMES[0], True),
            # This is the configuration that previously underallocated the CC
            # reflux buffer when direct drift correction was enabled.
            run_case(BASENAMES[1], False),
            run_case(BASENAMES[2], True, [
                "problem/p0=0.1",
                "problem/compression_rate_x1=5.0",
                "mhd/dual_energy_eta1=0.1",
                "mhd/dual_energy_eta2=0.05",
                "time/tlim=0.001",
                "output2/dt=0.001", "output3/dt=0.001",
                "output4/dt=0.001", "output8/dt=0.001",
            ]),
            run_case(BASENAMES[3], True, [
                "problem/rho0=0.1",
                "problem/density_amplitude=0.0",
                "problem/pressure_amplitude=0.0",
                "problem/checkerboard_b3_amplitude=1.0",
                "mhd/sigma_max=2.0",
                "time/tlim=1.0e-8",
                "output2/dt=1.0e-8", "output3/dt=1.0e-8",
                "output4/dt=1.0e-8", "output8/dt=1.0e-8",
            ]),
        ]
        for result in results[:2]:
            assert_invariants(result)
        assert_invariants(results[2], expected_cycles=1)
        assert_invariants(results[3], expected_cycles=1)

        # In this benign smooth state, the auxiliary dual-energy fallback should not
        # materially change the physical multilevel solution.
        for key in ("dens", "eint", "bcc3"):
            assert relative_l2(results[0][0][key], results[1][0][key]) < 1.0e-3
        for key in ("eion", "eele"):
            assert relative_l2(results[0][1][key], results[1][1][key]) < 1.0e-3
        assert_dual_energy_branch_flip(results[2], 0.1)
        assert_magnetization_floor(results[3], 2.0)
    finally:
        testutils.cleanup()
        for basename in BASENAMES:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for path in Path("bin").glob(f"{basename}.*.bin"):
                path.unlink(missing_ok=True)
