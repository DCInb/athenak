"""Adaptive-limit and fail-fast stress coverage for Biermann subcycling."""

from pathlib import Path
import re
import subprocess

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
DIAGNOSTIC = re.compile(
    rf"cycle=(?P<cycle>\d+).*?dt=(?P<dt>{FLOAT}).*?"
    rf"biermann_substeps=(?P<steps>\d+).*?"
    rf"biermann_dt_min=(?P<dt_min>{FLOAT}).*?"
    rf"biermann_dt_max=(?P<dt_max>{FLOAT}).*?"
    rf"biermann_interval=(?P<interval>{FLOAT}).*?"
    rf"biermann_max_ratio=(?P<ratio>{FLOAT})"
)


def common_flags(basename):
    return [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=16", "meshblock/nx2=16",
        "output2/slice_x2=0.015625",
        "output3/slice_x2=0.015625",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=20.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/tlim=1.0", "time/ndiag=1",
    ]


def run_capture(flags):
    return subprocess.run(
        ["./athena", "-i", INPUT_FILE, *flags],
        capture_output=True, text=True, timeout=120, check=False)


def test_run():
    basenames = (
        "biermann_adaptive_limit", "biermann_max_steps_guard",
        "biermann_masked", "biermann_unmasked",
        "biermann_checkerboard_control", "biermann_checkerboard_limit",
        "biermann_cfl_guard",
        "biermann_cfl_accuracy_015", "biermann_cfl_accuracy_0075",
    )
    try:
        # Two deliberately strong macrosteps make the magnetic/current state evolve
        # enough that the stability limit must be recomputed, rather than cached once.
        adaptive = run_capture([
            *common_flags(basenames[0]),
            "time/nlim=2",
            "output1/dt=0.01",
            "output2/dt=0.01", "output2/data_format=%24.17e",
            "output3/dt=0.01", "output3/data_format=%24.17e",
            "output4/dt=-1",
        ])
        assert adaptive.returncode == 0, adaptive.stdout + adaptive.stderr
        records = []
        for match in DIAGNOSTIC.finditer(adaptive.stdout + adaptive.stderr):
            record = {key: float(value) for key, value in match.groupdict().items()}
            record["cycle"] = int(record["cycle"])
            record["steps"] = int(record["steps"])
            records.append(record)
        assert [record["cycle"] for record in records] == [0, 1, 2], records
        first, second = records[1], records[2]
        assert first["steps"] >= 20
        assert second["steps"] > 1.5*first["steps"]
        assert first["dt_min"] < 0.6*first["dt_max"]
        assert second["dt_min"] < 0.8*first["dt_min"]
        assert second["dt_max"] > second["dt_min"]
        assert np.isclose(first["interval"], records[0]["dt"], rtol=2.0e-6)
        assert np.isclose(second["interval"], first["dt"], rtol=2.0e-6)
        for record in (first, second):
            assert 0.0 < record["ratio"] <= 1.0 + 2.0e-6

        field_path = sorted(Path("tab").glob(
            f"{basenames[0]}.biermann.*.tab"))[-1]
        two_temperature_path = sorted(Path("tab").glob(
            f"{basenames[0]}.two_temperature.*.tab"))[-1]
        field = athena_read.tab(field_path)
        two_temperature = athena_read.tab(two_temperature_path)
        for key in ("dens", "eint"):
            assert np.all(np.isfinite(field[key]))
            assert np.all(field[key] > 0.0)
        for key in ("eion", "eele", "tion", "tele"):
            assert np.all(np.isfinite(two_temperature[key]))
            assert np.all(two_temperature[key] > 0.0)
        assert np.max(np.abs(field["bcc3"])) > 1.0e-2
        component_sum = field["dens"]*(
            two_temperature["eion"] + two_temperature["eele"])
        assert np.allclose(component_sum, field["eint"],
                           rtol=3.0e-11, atol=3.0e-12)
        history = athena_read.hst(f"{basenames[0]}.mhd.hst")
        assert np.allclose(history["tot-E"], history["tot-E"][0],
                           rtol=3.0e-12, atol=3.0e-12)

        # A pathologically small cap must terminate with a useful diagnostic instead
        # of hanging or silently truncating the Strang half-interval.
        guarded = run_capture([
            *common_flags(basenames[1]),
            "mhd/biermann_subcycle_max_steps=1",
            "time/nlim=1",
            "output1/dt=-1", "output2/dt=-1",
            "output3/dt=-1", "output4/dt=-1",
        ])
        output = guarded.stdout + guarded.stderr
        assert guarded.returncode > 0, output
        assert "FATAL ERROR" in output
        assert "requires more than 1 microsteps" in output
        assert "remaining=" in output and "stability_limit=" in output

        # Values above the stability-qualified method factor must fail at input
        # validation instead of silently entering the known inaccurate regime.
        cfl_guarded = run_capture([
            *common_flags(basenames[6]),
            "mhd/biermann_subcycle_cfl=0.3",
            "time/nlim=1",
            "output1/dt=-1", "output2/dt=-1",
            "output3/dt=-1", "output4/dt=-1",
        ])
        output = cfl_guarded.stdout + cfl_guarded.stderr
        assert cfl_guarded.returncode > 0, output
        assert "FATAL ERROR" in output
        assert "biermann_subcycle_cfl must be finite and in (0,0.15]" in output

        # The production edge cochain replaces directional shock masking inside the
        # dedicated B operator.  A deliberately under-resolved compressive profile
        # activates the legacy detector, but toggling that detector must not change any
        # subcycled state: applying a spatial mask to an edge field would reintroduce
        # the nonconvergent mask-gradient curl that the cochain removes.
        treatment_states = {}
        for basename, suppress in zip(basenames[2:], ("true", "false")):
            masked_run = run_capture([
                *common_flags(basename),
                # Keep the periodic velocity reset at a pressure extremum, where
                # dp_e/dx vanishes.  The interior remains a uniformly converging
                # crossed-gradient fixture without an artificial expansion seam.
                "mesh/x1min=-0.25", "mesh/x1max=0.75",
                "problem/pressure_amplitude=2.0",
                "problem/compression_rate_x1=4.0",
                "problem/compression_rate_x2=0.0",
                "problem/compression_rate_x3=0.0",
                "mhd/biermann_coefficient=5.0",
                "mhd/biermann_shock_threshold=0.1",
                f"mhd/biermann_shock_suppression={suppress}",
                "time/nlim=-1", "time/tlim=0.001",
                "time/align_outputs=true",
                "output1/dt=0.001",
                "output2/dt=0.001", "output2/data_format=%24.17e",
                "output3/dt=0.001", "output3/data_format=%24.17e",
                "output4/dt=-1",
            ])
            assert masked_run.returncode == 0, (
                masked_run.stdout + masked_run.stderr)
            field = athena_read.tab(sorted(Path("tab").glob(
                f"{basename}.biermann.*.tab"))[-1])
            two_temperature = athena_read.tab(sorted(Path("tab").glob(
                f"{basename}.two_temperature.*.tab"))[-1])
            history = athena_read.hst(f"{basename}.mhd.hst")
            treatment_states[basename] = (field, two_temperature, history)
            for key in ("eion", "eele", "tion", "tele"):
                assert np.all(np.isfinite(two_temperature[key]))
                assert np.all(two_temperature[key] > 0.0)
            assert np.allclose(history["tot-E"], history["tot-E"][0],
                               rtol=3.0e-12, atol=3.0e-12)
        masked_state = treatment_states[basenames[2]]
        unmasked_state = treatment_states[basenames[3]]
        for masked_table, unmasked_table in zip(masked_state, unmasked_state):
            assert masked_table.keys() == unmasked_table.keys()
            for key in masked_table:
                assert np.array_equal(masked_table[key], unmasked_table[key]), key

        # A centered two-cell curl is blind to a Nyquist checkerboard.  The timestep
        # bound must nevertheless see its adjacent one-sided slopes and tile the
        # macro interval with real microsteps.  The zero-field case is the matched
        # control and needs exactly one microstep per Strang half.
        checkerboard_records = {}
        for basename, amplitude in zip(basenames[4:], (0.0, 0.25)):
            checkerboard = run_capture([
                *common_flags(basename),
                "problem/pressure_amplitude=0.0",
                "problem/density_amplitude=0.0",
                f"problem/checkerboard_b3_amplitude={amplitude}",
                "mhd/biermann_coefficient=2.0",
                "time/nlim=1", "time/tlim=1.0",
                "output1/dt=-1", "output2/dt=-1",
                "output3/dt=-1", "output4/dt=-1",
            ])
            assert checkerboard.returncode == 0, (
                checkerboard.stdout + checkerboard.stderr)
            records = list(DIAGNOSTIC.finditer(
                checkerboard.stdout + checkerboard.stderr))
            assert len(records) == 2, records
            checkerboard_records[basename] = {
                key: float(value)
                for key, value in records[-1].groupdict().items()
            }
        control = checkerboard_records[basenames[4]]
        limited = checkerboard_records[basenames[5]]
        assert int(control["steps"]) == 2, control
        assert int(limited["steps"]) >= 8, limited
        assert limited["dt_max"] < 0.25*limited["interval"], limited
        assert 0.0 < limited["ratio"] <= 1.0 + 2.0e-6, limited

        # Reproducibly qualify the largest accepted method factor against a tighter
        # integration of the same endpoint-cochain operator.  This nonlinear run is
        # long enough for magnetic feedback and electron work to affect the state;
        # it is not merely a one-step source comparison.
        cfl_states = {}
        for basename, cfl in zip(basenames[7:], (0.15, 0.075)):
            cfl_run = run_capture([
                f"job/basename={basename}",
                "mesh/nx1=128", "mesh/nx2=128",
                "meshblock/nx1=16", "meshblock/nx2=16",
                "mhd/biermann_shock_suppression=false",
                "mhd/biermann_coefficient=5.0",
                "mhd/biermann_subcycle=true",
                f"mhd/biermann_subcycle_cfl={cfl}",
                "time/tlim=0.02", "time/align_outputs=true",
                "output1/dt=0.02", "output2/dt=-1", "output3/dt=-1",
                "output4/dt=0.02", "output5/dt=0.02",
                "output6/dt=-1", "output7/dt=-1",
            ])
            assert cfl_run.returncode == 0, cfl_run.stdout + cfl_run.stderr
            state = bin_convert.read_binary(
                f"bin/{basename}.biermann_full.00001.bin")
            divergence = bin_convert.read_binary(
                f"bin/{basename}.divb.00001.bin")
            cfl_states[basename] = state

            def flatten(output, variable):
                return np.concatenate([
                    np.asarray(block).ravel()
                    for block in output["mb_data"][variable]
                ])

            density = flatten(state, "dens")
            internal = flatten(state, "eint")
            eion = flatten(state, "eion")
            eele = flatten(state, "eele")
            assert np.all(density > 0.0)
            assert np.all(internal > 0.0)
            assert np.all(eion > 0.0)
            assert np.all(eele > 0.0)
            assert np.allclose(density*(eion+eele), internal,
                               rtol=3.0e-11, atol=3.0e-12)
            divb = flatten(divergence, "divb")
            assert np.max(np.abs(divb)) < 5.0e-13
            cfl_history = athena_read.hst(f"{basename}.mhd.hst")
            assert np.allclose(cfl_history["tot-E"], cfl_history["tot-E"][0],
                               rtol=3.0e-12, atol=3.0e-12)

        upper = cfl_states[basenames[7]]
        tight = cfl_states[basenames[8]]
        assert upper["time"] == tight["time"] == 0.02
        assert upper["cycle"] == tight["cycle"]
        upper_b3 = flatten(upper, "bcc3")
        tight_b3 = flatten(tight, "bcc3")
        relative_b3 = np.linalg.norm(upper_b3-tight_b3)/np.linalg.norm(tight_b3)
        assert relative_b3 < 1.0e-4, relative_b3
        upper_me = 0.5*np.dot(upper_b3, upper_b3)
        tight_me = 0.5*np.dot(tight_b3, tight_b3)
        assert abs(upper_me/tight_me-1.0) < 2.0e-4
    finally:
        testutils.cleanup()
        for basename in basenames:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
