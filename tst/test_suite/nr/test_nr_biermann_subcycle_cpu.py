"""CPU regression coverage for second-order multirate Biermann integration."""

from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
FINAL_TIME = 2.0e-2
RESOLUTION = 32
BLOCK_SIZE = 16
GENERATED_BASENAMES = []


def run_case(basename, flags=None, final_time=FINAL_TIME, one_cycle=False):
    """Run the smooth periodic battery problem on a small, fixed spatial mesh."""
    Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
    GENERATED_BASENAMES.append(basename)
    run_flags = [
        f"job/basename={basename}",
        f"mesh/nx1={RESOLUTION}",
        f"mesh/nx2={RESOLUTION}",
        f"meshblock/nx1={BLOCK_SIZE}",
        f"meshblock/nx2={BLOCK_SIZE}",
        "output2/slice_x2=0.015625",
        "output3/slice_x2=0.015625",
        f"time/tlim={final_time}",
        f"output1/dt={final_time}",
        f"output2/dt={final_time}",
        f"output3/dt={final_time}",
    ]
    if one_cycle:
        run_flags.extend(["time/nlim=1", "time/tlim=1.0"])
    if flags:
        run_flags.extend(flags)
    assert testutils.run(INPUT_FILE, flags=run_flags, timeout=120.0), (
        f"{basename} run failed")


def sorted_table(path):
    """Read a 1-D tab slice and put cells in physical x1 order."""
    table = athena_read.tab(path)
    order = np.argsort(table["x1v"])
    sorted_data = {}
    for key, value in table.items():
        array = np.asarray(value)
        sorted_data[key] = array[order] if array.shape == order.shape else array
    return sorted_data


def load_final_state(basename):
    field = sorted_table(f"tab/{basename}.biermann.00001.tab")
    two_temperature = sorted_table(
        f"tab/{basename}.two_temperature.00001.tab")
    return field, two_temperature


def history(basename):
    return athena_read.hst(f"{basename}.mhd.hst")


def assert_energy_conserved(basename):
    data = history(basename)
    assert np.all(np.isfinite(data["tot-E"]))
    assert np.allclose(data["tot-E"], data["tot-E"][0],
                       rtol=3.0e-12, atol=3.0e-12)


def assert_component_closure(state):
    """Require the redundant 2T components to match conservative internal energy."""
    field, two_temperature = state
    component_density = np.asarray(field["dens"])*(
        np.asarray(two_temperature["eion"])
        + np.asarray(two_temperature["eele"]))
    assert np.allclose(component_density, np.asarray(field["eint"]),
                       rtol=3.0e-11, atol=3.0e-12)


def assert_tables_identical(left, right):
    """Require exact CPU output identity for an opt-in feature's disabled path."""
    assert left.keys() == right.keys()
    for key in left:
        assert np.array_equal(left[key], right[key]), (
            f"default and explicit-false outputs differ in {key}")


def relative_l2(left, right):
    denominator = np.linalg.norm(right)
    assert denominator > 0.0
    return np.linalg.norm(left-right)/denominator


def complete_biermann_components(state):
    """Return fields advanced by the CT, conservative-flux, and work operators."""
    field, two_temperature = state
    return {
        "bcc3": np.asarray(field["bcc3"]),
        "eint": np.asarray(field["eint"]),
        "eion": np.asarray(two_temperature["eion"]),
        "eele": np.asarray(two_temperature["eele"]),
    }


def normalized_state_distance(left, right, scale_state):
    """Dimensionless L2 distance giving each evolved component equal weight."""
    left_components = complete_biermann_components(left)
    right_components = complete_biermann_components(right)
    scale_components = complete_biermann_components(scale_state)
    errors = []
    for key in scale_components:
        scale = np.linalg.norm(scale_components[key])
        assert scale > 0.0
        errors.append(
            np.linalg.norm(left_components[key]-right_components[key])/scale)
    return np.linalg.norm(errors)/np.sqrt(len(errors))


def test_run():
    try:
        # The feature is opt-in: the deck default must execute exactly the same CPU
        # path as spelling out biermann_subcycle=false on the command line.
        default_flags = [
            "time/tlim=1.0e-4",
            "output1/dt=1.0e-4",
            "output2/dt=1.0e-4",
            "output3/dt=1.0e-4",
        ]
        run_case("biermann_subcycle_default", default_flags,
                 final_time=1.0e-4)
        run_case("biermann_subcycle_disabled", [
            *default_flags, "mhd/biermann_subcycle=false",
        ], final_time=1.0e-4)
        default_state = load_final_state("biermann_subcycle_default")
        disabled_state = load_final_state("biermann_subcycle_disabled")
        for default_table, disabled_table in zip(default_state, disabled_state):
            assert_tables_identical(default_table, disabled_table)
        assert_tables_identical(history("biermann_subcycle_default"),
                                history("biermann_subcycle_disabled"))

        # A configured subcycle with a zero coefficient is a strict no-op, including
        # its macro dual-energy path.  This guards the C_B -> 0 control used to isolate
        # any later production differences.
        zero_flags = [
            "mhd/biermann_coefficient=0.0",
            "time/tlim=1.0e-4",
            "output1/dt=1.0e-4",
            "output2/dt=1.0e-4",
            "output3/dt=1.0e-4",
        ]
        run_case("biermann_subcycle_zero_disabled", [
            *zero_flags, "mhd/biermann_subcycle=false",
        ], final_time=1.0e-4)
        run_case("biermann_subcycle_zero_enabled", [
            *zero_flags, "mhd/biermann_subcycle=true",
        ], final_time=1.0e-4)
        for disabled_table, enabled_table in zip(
                load_final_state("biermann_subcycle_zero_disabled"),
                load_final_state("biermann_subcycle_zero_enabled")):
            assert_tables_identical(disabled_table, enabled_table)
        assert_tables_identical(history("biermann_subcycle_zero_disabled"),
                                history("biermann_subcycle_zero_enabled"))

        # With a strong battery, legacy integration is battery-CFL limited.  In
        # multirate mode its first macro step must instead equal the battery-free
        # MHD step.  The large scale separation also forces many real microsteps.
        common_dt_flags = [
            "mhd/biermann_shock_suppression=false",
            "output1/dt=1.0", "output2/dt=1.0", "output3/dt=1.0",
        ]
        run_case("biermann_subcycle_dt_legacy", [
            *common_dt_flags,
            "mhd/biermann_coefficient=20.0",
            "mhd/biermann_subcycle=false",
        ], one_cycle=True)
        run_case("biermann_subcycle_dt_control", [
            *common_dt_flags,
            "mhd/biermann_coefficient=0.0",
            "mhd/biermann_subcycle=false",
        ], one_cycle=True)
        run_case("biermann_subcycle_dt_multirate", [
            *common_dt_flags,
            "mhd/biermann_coefficient=20.0",
            "mhd/biermann_subcycle=true",
            "mhd/biermann_subcycle_cfl=0.15",
        ], one_cycle=True)
        legacy_dt = history("biermann_subcycle_dt_legacy")["dt"][0]
        control_dt = history("biermann_subcycle_dt_control")["dt"][0]
        multirate_dt = history("biermann_subcycle_dt_multirate")["dt"][0]
        assert legacy_dt < 0.1*control_dt
        assert np.isclose(multirate_dt, control_dt, rtol=2.0e-13, atol=0.0)
        multirate_field, multirate_two_temperature = load_final_state(
            "biermann_subcycle_dt_multirate")
        assert np.max(np.abs(multirate_field["bcc3"])) > 1.0e-2
        for key in ("eion", "eele", "tion", "tele"):
            assert np.all(np.isfinite(multirate_two_temperature[key]))
            assert np.all(multirate_two_temperature[key] > 0.0)
        assert_component_closure(
            (multirate_field, multirate_two_temperature))
        assert_energy_conserved("biermann_subcycle_dt_multirate")

        # Use exact nested macro steps and scale the battery CFL with them.  CFL-
        # selected steps have state-dependent terminal remainders, whose non-nested
        # microstep partitions make adjacent Richardson ratios unnecessarily noisy.
        # History output alignment supplies deterministic macro steps while retaining
        # a genuine multirate separation in every run.
        subcycle_states = []
        refinement = zip(
            (0.15, 0.075, 0.0375, 0.01875, 0.009375,
             0.0046875, 0.00234375),
            (4, 8, 16, 32, 64, 128, 256),
        )
        for index, (cfl, macro_steps) in enumerate(refinement):
            basename = f"biermann_subcycle_convergence_{index}"
            macro_dt = FINAL_TIME/macro_steps
            run_case(basename, [
                "mhd/biermann_shock_suppression=false",
                "mhd/biermann_coefficient=5.0",
                "mhd/biermann_subcycle=true",
                f"mhd/biermann_subcycle_cfl={cfl}",
                "time/cfl_number=0.3",
                "time/align_outputs=true",
                f"output1/dt={macro_dt:.17g}",
            ])
            state = load_final_state(basename)
            subcycle_states.append(state)
            run_history = history(basename)
            assert len(run_history["time"])-1 == macro_steps
            assert_energy_conserved(basename)
            for key in ("eion", "eele", "tion", "tele"):
                assert np.all(np.isfinite(state[1][key]))
                assert np.all(state[1][key] > 0.0)
            assert_component_closure(state)

        # Use adjacent nested differences rather than treating the finest finite step
        # as an exact solution.  The latter biases rates upward and can make a genuinely
        # first-order path appear to pass.  Small cancellation oscillations are expected,
        # so require the final rate above 1.6 and the last two rates' mean above 1.8.
        finest_state = subcycle_states[-1]
        differences = [
            normalized_state_distance(
                subcycle_states[index], subcycle_states[index+1], finest_state)
            for index in range(len(subcycle_states)-1)
        ]
        assert np.all(np.asarray(differences) > 0.0)
        observed_orders = np.log2(
            np.asarray(differences[:-1])/np.asarray(differences[1:]))
        assert observed_orders[-1] >= 1.6, observed_orders
        assert np.mean(observed_orders[-2:]) >= 1.8, observed_orders

        # The aggregate norm must not hide a lower-order piece of the split
        # operator.  In particular, checking eion separately catches a
        # first-order electron-work/dual-energy synchronization.
        component_states = [
            complete_biermann_components(state) for state in subcycle_states
        ]
        for key in component_states[0]:
            component_differences = [
                np.linalg.norm(component_states[index][key]
                               - component_states[index+1][key])
                for index in range(len(component_states)-1)
            ]
            assert np.all(np.asarray(component_differences) > 0.0), (
                key, component_differences)
            component_orders = np.log2(
                np.asarray(component_differences[:-1])
                / np.asarray(component_differences[1:]))
            assert component_orders[-1] >= 1.6, (
                key, component_orders)
            assert np.mean(component_orders[-2:]) >= 1.8, (
                key, component_orders)

        # A fine legacy calculation integrates the unsplit operator at the battery
        # timestep.  The production-CFL multirate result must agree in magnetic and
        # thermal energies while preserving the same global invariant.
        run_case("biermann_subcycle_fine_legacy", [
            "mhd/biermann_shock_suppression=false",
            "mhd/biermann_coefficient=5.0",
            "mhd/biermann_subcycle=false",
            "time/cfl_number=0.0375",
        ])
        legacy_state = load_final_state("biermann_subcycle_fine_legacy")
        coarse_components = complete_biermann_components(subcycle_states[0])
        legacy_components = complete_biermann_components(legacy_state)
        assert relative_l2(coarse_components["bcc3"],
                           legacy_components["bcc3"]) < 2.5e-3
        assert relative_l2(coarse_components["eint"],
                           legacy_components["eint"]) < 5.0e-4
        # The legacy stage projection distributes the finite-grid CT/energy closure
        # residual across both species, while the DAE subcycle preserves the physical
        # electron equation and assigns that truncation residual to redundant ion
        # energy.  Their component partitions therefore agree to spatial, not temporal,
        # truncation accuracy on this fixed mesh.
        for key in ("eion", "eele"):
            assert relative_l2(coarse_components[key],
                               legacy_components[key]) < 1.0e-3
        coarse_history = history("biermann_subcycle_convergence_0")
        fine_subcycle_history = history("biermann_subcycle_convergence_6")
        legacy_history = history("biermann_subcycle_fine_legacy")
        assert np.isclose(coarse_history["3-ME"][-1],
                          fine_subcycle_history["3-ME"][-1],
                          rtol=7.0e-4, atol=0.0)
        # Magnetic energy is quadratic in B, so its legacy discrepancy is about twice
        # the already-gated field-norm discrepancy on this fixed spatial mesh.
        assert np.isclose(coarse_history["3-ME"][-1],
                          legacy_history["3-ME"][-1],
                          rtol=7.5e-3, atol=0.0)
        assert_energy_conserved("biermann_subcycle_fine_legacy")
    finally:
        testutils.cleanup()
        for basename in GENERATED_BASENAMES:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
