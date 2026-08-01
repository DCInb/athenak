"""Null-curl and oblique-front coverage for the production Biermann edge cochain."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
FINAL_TIME = 1.0e-8
BASENAMES = (
    "biermann_cochain_null_2d",
    "biermann_cochain_aligned_front",
    "biermann_cochain_oblique_front",
    "biermann_cochain_null_3d",
    "biermann_cochain_null_smr_2d",
    "biermann_cochain_null_smr_3d",
    "biermann_cochain_null_outflow_smr_3d",
)

EXPECTED_REFINED_TOPOLOGY = {
    BASENAMES[4]: {0: 15, 1: 4},
    BASENAMES[5]: {0: 7, 1: 8},
    BASENAMES[6]: {0: 7, 1: 8},
}


def run_case(basename, resolution, pressure, compression, three_d=False,
             extra_flags=None):
    block = 8 if three_d else 16
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}", f"mesh/nx2={resolution}",
        f"meshblock/nx1={block}", f"meshblock/nx2={block}",
        "mesh/x1min=-0.25", "mesh/x1max=0.75",
        "mesh/x2min=-0.25", "mesh/x2max=0.75",
        "problem/density_amplitude=0.0",
        "problem/density_x3_amplitude=0.0",
        f"problem/pressure_amplitude={pressure[0]}",
        f"problem/pressure_x2_amplitude={pressure[1]}",
        f"problem/pressure_x3_amplitude={pressure[2]}",
        f"problem/compression_rate_x1={compression[0]}",
        f"problem/compression_rate_x2={compression[1]}",
        f"problem/compression_rate_x3={compression[2]}",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "mhd/biermann_shock_suppression=true",
        "mhd/biermann_shock_threshold=0.1",
        "mhd/biermann_shock_compression_threshold=0.0",
        f"time/tlim={FINAL_TIME}",
        "time/align_outputs=true",
        f"output1/dt={FINAL_TIME}",
        "output2/dt=-1", "output3/dt=-1",
        "output4/variable=mhd_divb", "output4/id=divb",
        "output4/double_precision_binary=true",
        f"output4/dt={FINAL_TIME}",
        f"output5/dt={FINAL_TIME}",
        "output6/dt=-1", "output7/dt=-1",
    ]
    if three_d:
        flags.extend([
            f"mesh/nx3={resolution}", f"meshblock/nx3={block}",
            "mesh/x3min=-0.25", "mesh/x3max=0.75",
        ])
    if extra_flags:
        flags.extend(extra_flags)
    assert testutils.run(INPUT_FILE, flags=flags, timeout=120.0), (
        f"{basename} run failed")


def magnetic_maxima(basename):
    output = bin_convert.read_binary(
        f"bin/{basename}.biermann_full.00001.bin")
    return np.array([
        np.max(np.abs(np.asarray(output["mb_data"][component])))
        for component in ("bcc1", "bcc2", "bcc3")
    ])


def assert_conservative(basename):
    history = athena_read.hst(f"{basename}.mhd.hst")
    assert len(history["time"]) == 2
    assert np.allclose(history["tot-E"], history["tot-E"][0],
                       rtol=3.0e-12, atol=3.0e-12)
    face_magnetic_energy = np.array([
        history[key][-1] for key in ("1-ME", "2-ME", "3-ME")
    ])
    assert np.all(np.isfinite(face_magnetic_energy)), (
        basename, face_magnetic_energy)
    assert np.array_equal(
        np.array([history[key][0] for key in ("1-ME", "2-ME", "3-ME")]),
        np.zeros(3)), basename
    # Cell-centred output can hide equal-and-opposite face errors.  Gate the
    # face-centred fields used by CT independently through their history energies.
    assert np.all(face_magnetic_energy < 2.0e-24), (
        basename, face_magnetic_energy)


def assert_divergence_free(basename):
    output = bin_convert.read_binary(f"bin/{basename}.divb.00001.bin")
    assert output["cycle"] == 1, (basename, output["cycle"])
    assert output["var_names"] == ["divb"], (
        basename, output["var_names"])
    divb = np.concatenate([
        np.asarray(block).ravel() for block in output["mb_data"]["divb"]
    ])
    assert np.all(np.isfinite(divb)), basename
    assert np.max(np.abs(divb)) < 5.0e-13, (
        basename, np.max(np.abs(divb)))
    return output


def assert_block_topology(basename, output):
    levels, counts = np.unique(
        np.asarray(output["mb_logical"])[:, 3], return_counts=True)
    topology = dict(zip(levels.tolist(), counts.tolist()))
    assert topology == EXPECTED_REFINED_TOPOLOGY[basename], (
        basename, topology)


def test_run():
    try:
        # Constant electron density makes -C*grad(p_e)/n_e an exact gradient.
        # The mixed pressure and one-axis compression activate the old directional
        # mask on only one component; that operator produced an O(1) nonconvergent
        # B3 source on this fixture.
        run_case(BASENAMES[0], 32, (2.0, 0.04, 0.0),
                 (1.0e-3, 0.0, 0.0))

        # A steep aligned front is the control.  Equal x/y amplitudes and
        # compression rotate the local front normal off the grid axes and activate
        # both directional shock sensors.  Neither may manufacture circulation.
        run_case(BASENAMES[1], 32, (4.0, 0.0, 0.0),
                 (1.0e-3, 0.0, 0.0))
        run_case(BASENAMES[2], 32, (4.0, 4.0, 0.0),
                 (1.0e-3, 1.0e-3, 0.0))

        # Exercise all three endpoint directions and every CT curl orientation.
        run_case(BASENAMES[3], 16, (2.0, 0.7, 0.3),
                 (1.0e-3, 1.0e-3, 1.0e-3), three_d=True)

        # Refine only one corner root block.  Independently offsetting tangential
        # fine-edge pairs at the two orthogonal mortars used to manufacture an O(1)
        # circulation here even though the resulting face field remained divergence
        # free.  A single composite vertex trace must retain the exact-gradient null.
        run_case(BASENAMES[4], 32, (2.0, 0.04, 0.0),
                 (1.0e-3, 0.0, 0.0), extra_flags=[
                     "meshblock/nx1=8", "meshblock/nx2=8",
                     "mesh_refinement/refinement=static",
                     "refined_region1/x1min=-0.25",
                     "refined_region1/x1max=-0.24",
                     "refined_region1/x2min=-0.25",
                     "refined_region1/x2max=-0.24",
                 ])

        # The same corner in true 3-D exercises all three edge orientations and
        # the eight fine blocks meeting the three orthogonal refinement faces.
        # Before the dimension-independent vertex mortar this exact-gradient
        # state generated O(1e-5) magnetic fields at the refinement corner.
        run_case(BASENAMES[5], 16, (2.0, 0.7, 0.3),
                 (1.0e-3, 1.0e-3, 1.0e-3), three_d=True,
                 extra_flags=[
                     "mesh_refinement/refinement=static",
                     "refined_region1/x1min=-0.25",
                     "refined_region1/x1max=-0.24",
                     "refined_region1/x2min=-0.25",
                     "refined_region1/x2max=-0.24",
                     "refined_region1/x3min=-0.25",
                     "refined_region1/x3max=-0.24",
                 ])

        # Put that refined corner on three physical faces.  Uniform electron
        # density still makes the Biermann electric field an exact gradient, and
        # zero velocity isolates the boundary/mortar cochain from physical fluxes.
        # All six outflow faces are intentional: this catches physical-boundary
        # edge ownership errors that a periodic corner cannot exercise.
        run_case(BASENAMES[6], 16, (2.0, 0.7, 0.3),
                 (0.0, 0.0, 0.0), three_d=True, extra_flags=[
                     "mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
                     "mesh/ix2_bc=outflow", "mesh/ox2_bc=outflow",
                     "mesh/ix3_bc=outflow", "mesh/ox3_bc=outflow",
                     "mesh_refinement/refinement=static",
                     "refined_region1/x1min=-0.25",
                     "refined_region1/x1max=-0.24",
                     "refined_region1/x2min=-0.25",
                     "refined_region1/x2max=-0.24",
                     "refined_region1/x3min=-0.25",
                     "refined_region1/x3max=-0.24",
                 ])

        maxima = {basename: magnetic_maxima(basename)
                  for basename in BASENAMES}
        divb_outputs = {
            basename: assert_divergence_free(basename)
            for basename in BASENAMES
        }
        for basename in EXPECTED_REFINED_TOPOLOGY:
            assert_block_topology(basename, divb_outputs[basename])
        for basename, values in maxima.items():
            assert np.all(np.isfinite(values)), (basename, values)
            assert_conservative(basename)
            assert np.max(values) < 2.0e-12, (basename, values)
    finally:
        testutils.cleanup()
        for basename in BASENAMES:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
