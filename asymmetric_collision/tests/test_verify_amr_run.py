"""Focused log-only checks; no solver, GPU, or large restart files required."""

import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "collision_amr_validator", Path(__file__).resolve().parents[1] / "scripts/verify_amr_run.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def log_for(dts, mixed_from=5, command_extra=""):
    """Last dt belongs to Finalize and has never been advanced."""
    nlim = len(dts)-1
    lines = [f"Command: mpirun -n 1 ./athena -i case.athinput time/ndiag=1 "
             f"time/nlim={nlim} {command_extra}"]
    time = 0.
    for cycle, dt in enumerate(dts):
        lines.append(f"elapsed={cycle:.6e} cycle={cycle} time={time:.6e} dt={dt:.6e}")
        if cycle < nlim:
            if cycle >= mixed_from:
                lines.append("# implicit radiation phase start: group=0 "
                             "solver=high_order_fgmres attempt=0 tolerance=1e-7")
            lines.append("# implicit thermal radiation: groups=20 max_iterations=10")
            time += dt
    lines.extend(["", "Terminating on cycle limit", f"time={time:.6e} cycle={nlim}",
                  f"tlim=8.000000e-01 nlim={nlim}", ""])
    return "\n".join(lines)


class NormalTimestepTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "time": {"initial_dt": "1e-9", "tlim": ".8", "nlim": "-1",
                     "ndiag": "10", "cfl_number": ".25", "align_outputs": "true"},
            "output1": {"dt": "1"}, "output6": {"dt": "1e-12", "dcycle": "5"}}
        self.growth = [1e-9*2**cycle for cycle in range(8)]
        self.normal = self.growth+[2e-7, 2e-7]

    def inspect(self, dts=None, log=None, parameters=None):
        dts = self.normal if dts is None else dts
        return VALIDATOR.inspect_normal_timesteps(
            log_for(dts) if log is None else log,
            self.parameters if parameters is None else parameters, len(dts)-1)

    def test_completed_normal_step_passes(self):
        result = self.inspect()
        self.assertEqual(result["normal_control_steps"], 1)
        self.assertEqual(result["first_refined_normal_step"]["cycle"], 8)
        self.assertEqual(result["first_refined_normal_step"]["completed_cycle"], 9)
        self.assertEqual(result["final_planned_dt_not_counted"], 2e-7)

    def test_old_eight_cycle_warmup_rejected(self):
        with self.assertRaisesRegex(ValueError, "No completed timestep"):
            self.inspect([1e-9*2**cycle for cycle in range(9)])

    def test_planned_next_normal_dt_is_not_completed(self):
        with self.assertRaisesRegex(ValueError, "No completed timestep"):
            self.inspect(self.growth+[2e-7])

    def test_later_completed_normal_steps_count(self):
        result = self.inspect(self.normal+[1.9e-7, 1.8e-7])
        self.assertEqual(result["normal_control_steps"], 3)
        self.assertEqual(result["last_refined_normal_step"]["completed_cycle"], 11)

    def test_near_doubling_is_ambiguous(self):
        dts = self.growth+[2.56e-7*(1.-1e-7), 5.12e-7*(1.-1e-7)]
        with self.assertRaisesRegex(ValueError, "No completed timestep"):
            self.inspect(dts)

    def test_final_refinement_without_refined_normal_step_rejected(self):
        log = log_for(self.normal, mixed_from=9)
        log += "Current number of MeshBlocks = 250\n"
        with self.assertRaisesRegex(ValueError, "mixed-level solver evidence"):
            self.inspect(log=log)

    def test_earlier_mixed_step_does_not_certify_normal_step(self):
        log = log_for(self.normal, mixed_from=9)
        log = log.replace("elapsed=5.000000e+00", "# implicit radiation AMR fallback: "
                          "group=2 reason=negative_energy minimum=-1e-19\nelapsed=5.000000e+00")
        with self.assertRaisesRegex(ValueError, "mixed-level solver evidence"):
            self.inspect(log=log)

    def test_old_and_new_mixed_solver_markers(self):
        old = ("# implicit radiation phase start: group=0 "
               "solver=high_order_fgmres attempt=0 tolerance=1e-7")
        for marker in (
            "# implicit radiation AMR fallback: group=2 reason=negative_energy minimum=-1e-19",
            "# implicit radiation AMR GMRES finished: iterations=12 status=converged",
            "# implicit radiation Krylov: solver=FGMRES status=complete outer_iterations=12",
            "# implicit radiation phase complete: group=0 solver=high_order_gmres status=pass",
        ):
            with self.subTest(marker=marker):
                self.assertEqual(self.inspect(log=log_for(self.normal).replace(old, marker))[
                    "refined_normal_control_steps"], 1)

    def test_restart_launch_rejected_even_with_synthetic_zero_cycle(self):
        with self.assertRaisesRegex(ValueError, "fresh evolution"):
            self.inspect(log=log_for(self.normal, command_extra="-r previous.rst"))

    def test_tlim_clipping_rejected(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["time"]["tlim"] = str(sum(self.normal[:-1]))
        with self.assertRaisesRegex(ValueError, "Terminal time clipping"):
            self.inspect(parameters=parameters)

    def test_aligned_output_clipping_rejected(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["output1"]["dt"] = "4e-7"
        with self.assertRaisesRegex(ValueError, "Aligned output1"):
            self.inspect(parameters=parameters)

    def test_launch_output_override_is_observed(self):
        with self.assertRaisesRegex(ValueError, "Aligned output1"):
            self.inspect(log=log_for(self.normal, command_extra="output1/dt=4e-7"))

    def test_unaligned_outputs_do_not_shorten_dt(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["time"]["align_outputs"] = "false"
        parameters["output1"]["dt"] = "4e-7"
        self.assertEqual(self.inspect(parameters=parameters)["status"], "passed")

    def test_cycle_output_overrides_time_output(self):
        self.assertEqual(self.inspect()["status"], "passed")

    def test_nonfinite_or_nonpositive_diagnostics_rejected(self):
        for old, new in (("elapsed=1.000000e+00", "elapsed=nan"),
                         ("time=1.000000e-09", "time=inf"),
                         ("dt=2.000000e-09", "dt=nan"),
                         ("dt=2.000000e-09", "dt=0"),
                         ("dt=2.000000e-09", "dt=-1e-9")):
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.inspect(log=log_for(self.normal).replace(old, new, 1))

    def test_nonfinite_controls_rejected(self):
        for key in ("initial_dt", "tlim", "cfl_number"):
            parameters = copy.deepcopy(self.parameters)
            parameters["time"][key] = "nan"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.inspect(parameters=parameters)

    def test_missing_duplicate_and_reordered_cycles_rejected(self):
        for old, new in (("cycle=4 time=", "cycle=5 time="),
                         ("cycle=4 time=", "cycle=3 time="),
                         ("cycle=0 time=", "cycle=1 time=")):
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, "contiguous"):
                self.inspect(log=log_for(self.normal).replace(old, new, 1))

    def test_wrong_advance_rejected(self):
        with self.assertRaisesRegex(ValueError, "logged timestep"):
            self.inspect(log=log_for(self.normal).replace(
                "time=3.000000e-09", "time=3.100000e-09"))

    def test_elapsed_time_reversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonincreasing"):
            self.inspect(log=log_for(self.normal).replace(
                "elapsed=4.000000e+00", "elapsed=1.000000e+00"))

    def test_growth_above_two_rejected(self):
        dts = self.growth+[3e-7, 3e-7]
        with self.assertRaisesRegex(ValueError, "2x growth limit"):
            self.inspect(dts)

    def test_walltime_stop_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle limit"):
            self.inspect(log=log_for(self.normal).replace(
                "Terminating on cycle limit", "Terminating on wall clock limit"))

    def test_missing_finalize_row_rejected(self):
        log = log_for(self.normal)
        log = "\n".join(line for line in log.splitlines() if not line.startswith(
            "elapsed=9.000000e+00"))+"\n"
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.inspect(log=log)

    def test_ndiag_must_be_one(self):
        with self.assertRaisesRegex(ValueError, "ndiag=1"):
            self.inspect(log=log_for(self.normal, command_extra="time/ndiag=2"))

    def test_default_verify_signature_remains_compatible(self):
        self.assertEqual(VALIDATOR.verify.__defaults__, (8, 10, False))


if __name__ == "__main__":
    unittest.main()
