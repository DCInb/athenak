"""Checkpoint-start validation contracts without a solver or GPU launch."""

import copy
from pathlib import Path
import shlex
import shutil
import struct
import tempfile
import unittest

import numpy as np

from test_verify_amr_run import VALIDATOR


def parameter_text(parameters):
    return ''.join(f'<{block}>\n'+''.join(f'{key} = {value}\n' for key, value in values.items())
                   for block, values in parameters.items())


def segment_log(dts=(1.e-6, 2.e-6, 3.e-6, 3.e-6), start_cycle=10, start_time=1.e-6):
    end = start_cycle+len(dts)-1
    lines = [f'Command: ./athena -r start/checkpoint.rst time/ndiag=1 time/nlim={end}']
    time = start_time
    for offset, dt in enumerate(dts):
        cycle = start_cycle+offset
        lines.append(f'elapsed={offset:.6e} cycle={cycle} time={time:.6e} dt={dt:.6e}')
        if offset < len(dts)-1:
            lines.append('# implicit radiation AMR GMRES finished: iterations=5 status=converged')
            time += dt
    lines.extend(['', 'Terminating on cycle limit', f'time={time:.6e} cycle={end}', ''])
    return '\n'.join(lines)


class SegmentTimestepTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            'time': {'initial_dt': '1e-9', 'tlim': '.8', 'nlim': '13', 'ndiag': '1',
                     'cfl_number': '.25', 'align_outputs': 'true',
                     'restart_growth_cycle': '10', 'restart_growth_dt': '5e-7',
                     'restart_growth_terminal': '0'},
            'output1': {'dt': '1', 'last_time': '0'},
            'output2': {'dt': '-1'}, 'output6': {'dt': '-1', 'dcycle': '5'}}
        self.start = {'source_state': {'cycle': 10, 'time': 1e-6},
                      'parameters': self.parameters, 'growth_dt': 5e-7}

    def inspect(self, log=None):
        return VALIDATOR.inspect_normal_timesteps(
            segment_log() if log is None else log, self.parameters, 13, self.start)

    def test_counts_only_completed_candidate_segment(self):
        result = self.inspect()
        self.assertEqual(result['completed_steps'], 3)
        self.assertEqual(result['start_cycle'], 10)
        self.assertEqual(result['first_refined_normal_step']['cycle'], 12)
        self.assertEqual(result['mode'], 'checkpoint-start')

    def test_unknown_restart_is_still_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, 'fresh evolution'):
            VALIDATOR.inspect_normal_timesteps(segment_log(), self.parameters, 13)

    def test_first_restarted_step_never_certifies_normal_control(self):
        log = segment_log((2e-7, 4e-7, 8e-7, 1.6e-6))
        with self.assertRaisesRegex(ValueError, 'No completed timestep'):
            self.inspect(log)

    def test_wrong_start_time_and_cycle_rejected(self):
        for log in (segment_log(start_time=2e-6), segment_log(start_cycle=9)):
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.inspect(log)

    def test_missing_stale_nonfinite_or_terminal_growth_rejected(self):
        for key, value in (('restart_growth_cycle', '9'), ('restart_growth_dt', 'nan'),
                           ('restart_growth_dt', '0'), ('restart_growth_terminal', '1')):
            with self.subTest(key=key):
                original = self.parameters['time'][key]
                self.parameters['time'][key] = value
                with self.assertRaises(ValueError):
                    self.inspect()
                self.parameters['time'][key] = original

    def test_first_step_cannot_exceed_saved_growth_limit(self):
        with self.assertRaisesRegex(ValueError, 'saved 2x growth'):
            self.inspect(segment_log((2e-6, 3e-6, 4e-6, 4e-6)))

    def test_restarted_output_schedule_cannot_mimic_normal_control(self):
        self.parameters['output1'].update(dt='3e-6', last_time='1e-6')
        with self.assertRaisesRegex(ValueError, 'Aligned output1'):
            self.inspect()

    def test_missing_positive_output_schedule_rejected(self):
        del self.parameters['output1']['last_time']
        with self.assertRaisesRegex(ValueError, 'Nonfinite output1/last_time'):
            self.inspect()


class FrozenCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root/'validation'
        self.directory.mkdir()
        self.source = self.root/'source/rst/source.rst'
        self.source.parent.mkdir(parents=True)
        self.parameters = {
            'mesh': {**{f'nx{i}': '2' for i in (1, 2, 3)},
                     **{f'x{i}min': '-.5' for i in (1, 2, 3)},
                     **{f'x{i}max': '.5' for i in (1, 2, 3)}},
            'meshblock': {f'nx{i}': '2' for i in (1, 2, 3)},
            'mesh_refinement': {'refinement': 'adaptive', 'num_levels': '2',
                                'max_nmb_per_rank': '432', 'restart_age_version': '1'},
            'hydro': {'two_temperature': 'true', 'nscalars': '5'},
            'thermal_radiation': {'n_groups': '20', 'implicit_preconditioner': 'block-coarse',
                                  'implicit_tolerance': '1e-7'},
            'units': {'mass_cgs': '1', 'length_cgs': '1', 'time_cgs': '1'},
            'time': {'align_outputs': 'true', 'initial_dt': '1e-9', 'tlim': '.8',
                     'cfl_number': '.25', 'nlim': '25', 'ndiag': '1',
                     'restart_growth_cycle': '10', 'restart_growth_dt': '5e-7',
                     'restart_growth_terminal': '0'},
            'output1': {'dt': '1', 'last_time': '0'}}
        self.reference = copy.deepcopy(self.parameters)
        self.reference['thermal_radiation']['implicit_preconditioner'] = 'amr-aggregate'
        self.write_reference()
        self.write_checkpoint()
        executable = self.root/'source/athena'
        executable.write_bytes(b'frozen historical executable fixture\n')
        (self.root/'source/run.log').write_text(
            f'Backend: cuda\nExecutable SHA-256: {VALIDATOR.file_sha256(executable)}  '
            f'{executable}\n')
        self.tables = Path(__file__).resolve().parents[1] / 'material_tables.sha256'
        shutil.copy2(self.tables, self.root/'source/material_tables.sha256')

    def write_reference(self):
        (self.directory/'asymmetric_collision_amr.athinput').write_text(
            parameter_text(self.reference))

    def write_checkpoint(self, negative=False):
        state = np.zeros((1, 32, 4, 4, 4), dtype=np.float64)
        state[:, 0] = 1.
        state[:, 4] = 2.
        state[:, 5:7] = .5
        state[:, 7:9] = .2
        state[:, 9] = .1
        state[:, 10:12] = 1.
        state[:, 12:] = 1e-6
        if negative:
            state[0, 12, 1, 1, 1] = -1.
        indices = (1, 2, 2, 2)+15*(0,)
        with self.source.open('wb') as stream:
            stream.write((parameter_text(self.parameters)+'<par_end>\n').encode())
            stream.write(struct.pack('=ii9d', 1, 0, *(-.5,)*3, *(.5,)*6))
            stream.write(struct.pack('=19i', *indices)*2)
            stream.write(struct.pack('=ddi', 1e-6, 5e-7, 10))
            stream.write(struct.pack('=4ifQ', 0, 0, 0, 0, 1., state.nbytes))
            stream.write(state.tobytes())
            stream.write(struct.pack('=i', 10))

    def prepare(self):
        return VALIDATOR.prepare_start_checkpoint(self.directory, self.source, 25)

    def prepare_final_inspection(self):
        result = self.prepare()
        (self.directory/'initial').mkdir()
        shutil.copy2(self.tables, self.directory/'initial/material_tables.sha256')
        (self.directory/'evolve').mkdir()
        command = ['athena', '-r', str(self.directory/'start/checkpoint.rst')]
        command += [f'{key}={value}' for key, value in result['solver_overlays'].items()]
        (self.directory/'evolve/run.log').write_text('Command: '+shlex.join(command)+'\n')
        return result

    def test_real_restart_payload_freezes_and_rechecks(self):
        prepared = self.prepare_final_inspection()
        checked = VALIDATOR.inspect_start_checkpoint(
            self.directory, self.reference, 25, initial_parameters=self.reference)
        self.assertEqual(checked['source_state']['cycle'], 10)
        self.assertEqual(checked['checkpoint_sha256'], VALIDATOR.file_sha256(self.source))
        self.assertNotIn('thermal_radiation/implicit_gmres_metric', prepared['solver_overlays'])
        self.assertEqual(prepared['absent_solver_keys_using_constructor_defaults'][
            'thermal_radiation/implicit_gmres_metric'], 'volume')
        self.assertEqual(set(prepared['source_to_candidate_solver_changes']),
                         {'thermal_radiation/implicit_preconditioner'})

    def test_inherited_opacity_scale_missing_from_deck_is_rejected(self):
        self.parameters["materials"] = {"material0_opacity_value_scale": "2"}
        self.write_checkpoint()
        self.prepare_final_inspection()  # Raw-deck preflight cannot see this key.
        initial = copy.deepcopy(self.reference)
        initial["materials"] = {"material0_opacity_value_scale": "1"}
        with self.assertRaisesRegex(ValueError, "material0_opacity_value_scale"):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=initial)

    def test_inherited_source_tolerance_missing_from_deck_is_rejected(self):
        self.parameters["thermal_radiation"]["source_nonlinear_tolerance"] = "1e-3"
        self.write_checkpoint()
        self.prepare_final_inspection()
        initial = copy.deepcopy(self.reference)
        initial["thermal_radiation"]["source_nonlinear_tolerance"] = "1e-10"
        with self.assertRaisesRegex(ValueError, "source_nonlinear_tolerance"):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=initial)

    def test_negative_radiation_rejected_before_launch(self):
        self.write_checkpoint(negative=True)
        with self.assertRaisesRegex(ValueError, 'Negative ion, electron, or radiation'):
            self.prepare()

    def test_physics_controls_and_tolerance_cannot_be_overlaid(self):
        changes = (('time', 'cfl_number', '.2'), ('mesh', 'nx1', '4'),
                   ('thermal_radiation', 'implicit_tolerance', '1e-6'),
                   ('units', 'mass_cgs', '2'))
        for block, key, value in changes:
            altered = copy.deepcopy(self.reference)
            altered[block][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Changed'):
                VALIDATOR.compare_start_parameters(altered, self.parameters)

    def test_changed_frozen_checkpoint_rejected(self):
        self.prepare_final_inspection()
        with (self.directory/'start/checkpoint.rst').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=self.reference)

    def test_unrecorded_solver_overlay_rejected(self):
        self.prepare_final_inspection()
        log = self.directory/'evolve/run.log'
        log.write_text(log.read_text().replace(
            'implicit_preconditioner=amr-aggregate', 'implicit_preconditioner=jacobi'))
        with self.assertRaisesRegex(ValueError, 'recorded start override'):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=self.reference)

    def test_absent_nondefault_solver_control_fails_closed(self):
        self.reference['thermal_radiation']['implicit_amr_inner_norm'] = 'preconditioner'
        self.write_reference()
        with self.assertRaisesRegex(ValueError, 'Cannot override absent'):
            self.prepare()

    def test_unapproved_time_override_rejected(self):
        self.prepare_final_inspection()
        log = self.directory/'evolve/run.log'
        log.write_text(log.read_text().strip()+' time/restart_growth_dt=1\n')
        with self.assertRaisesRegex(ValueError, 'Unapproved'):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=self.reference)

    def test_full_deck_overlay_rejected(self):
        self.prepare_final_inspection()
        log = self.directory/'evolve/run.log'
        log.write_text(log.read_text().strip()+' -i different.athinput\n')
        with self.assertRaisesRegex(ValueError, 'without a full input overlay'):
            VALIDATOR.inspect_start_checkpoint(
                self.directory, self.reference, 25, initial_parameters=self.reference)

    def test_changed_source_executable_rejected(self):
        (self.root/'source/athena').write_bytes(b'changed executable')
        with self.assertRaisesRegex(ValueError, 'Source executable differs'):
            self.prepare()

    def test_changed_source_material_tables_rejected(self):
        (self.root/'source/material_tables.sha256').write_text('different table manifest\n')
        with self.assertRaisesRegex(ValueError, 'material-table manifest differs'):
            self.prepare()

    def test_missing_growth_metadata_rejected(self):
        del self.parameters['time']['restart_growth_dt']
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, 'growth dt'):
            self.prepare()

    def test_final_state_cannot_precede_start(self):
        with self.assertRaisesRegex(ValueError, 'below nlim'):
            VALIDATOR.prepare_start_checkpoint(self.directory, self.source, 10)


class ResolvedPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "hydro": {"sfloor": "1e-30"},
            "thermal_radiation": {"implicit_tolerance": "1e-7", "implicit_amr_order": "2"},
            "time": {"initial_dt": "1e-9", "tlim": ".8", "cfl_number": ".25"},
            "output1": {"dt": "1", "file_type": "bin"},
        }
        self.source = copy.deepcopy(self.initial)

    def test_source_only_physics_key_rejected(self):
        self.source["hydro"]["unrequested_physics"] = "1"
        with self.assertRaisesRegex(ValueError, "Missing resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_source_only_empty_block_rejected(self):
        self.source["unrequested_physics"] = {}
        with self.assertRaisesRegex(ValueError, "Missing resolved start block"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_candidate_only_physics_default_rejected(self):
        self.initial["hydro"]["new_physics_default"] = "1"
        with self.assertRaisesRegex(ValueError, "Missing resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_accuracy_and_timestep_controls_remain_strict(self):
        changes = (("thermal_radiation", "implicit_tolerance", "1e-6"),
                   ("thermal_radiation", "implicit_amr_order", "1"),
                   ("time", "initial_dt", "1e-8"), ("time", "tlim", "1"),
                   ("time", "cfl_number", ".3"))
        for block, key, value in changes:
            source = copy.deepcopy(self.source)
            source[block][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Changed resolved"):
                VALIDATOR.compare_resolved_start_physics(self.initial, source)

    def test_explicit_work_output_and_restart_metadata_allowed(self):
        self.source["thermal_radiation"]["implicit_max_iterations"] = "10000"
        self.source["time"].update(nlim="25", restart_growth_cycle="15",
                                   restart_growth_dt="1e-5", restart_growth_terminal="0")
        self.source["mesh_refinement"] = {"restart_age_version": "1"}
        self.source["output1"].update(dt="-1", last_time="0", file_number="3")
        result = VALIDATOR.compare_resolved_start_physics(self.initial, self.source)
        self.assertEqual(result["status"], "passed")
        self.assertIn("time/restart_growth_dt", result["allowed_differences"])

    def test_similarly_named_blocks_are_not_exempt(self):
        self.source["output_physics"] = {"dt": "1"}
        with self.assertRaisesRegex(ValueError, "Missing resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_solver_timers_are_diagnostic_metadata(self):
        self.initial["thermal_radiation"]["solver_timers"] = "true"
        result = VALIDATOR.compare_resolved_start_physics(self.initial, self.source)
        self.assertEqual(result["status"], "passed")
        self.assertIn("thermal_radiation/solver_timers", result["allowed_differences"])
        self.source["thermal_radiation"]["source_nonlinear_tolerance"] = "1e-4"
        with self.assertRaisesRegex(ValueError, "Missing resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_unlisted_output_key_remains_strict(self):
        self.source["output1"]["file_type"] = "rst"
        with self.assertRaisesRegex(ValueError, "Changed resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)

    def test_coarse_solver_backend_is_runtime_metadata(self):
        self.initial["thermal_radiation"]["implicit_amr_coarse_solver"] = "device"
        result = VALIDATOR.compare_resolved_start_physics(self.initial, self.source)
        self.assertEqual(result["status"], "passed")
        self.assertIn("thermal_radiation/implicit_amr_coarse_solver",
                      result["allowed_differences"])
        self.source["thermal_radiation"]["implicit_tolerance"] = "1e-4"
        with self.assertRaisesRegex(ValueError, "Changed resolved"):
            VALIDATOR.compare_resolved_start_physics(self.initial, self.source)


if __name__ == '__main__':
    unittest.main()
