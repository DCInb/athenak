"""Check Slurm script relocation, allocation sizing, and restart forwarding."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = self.root / 'case'
        scripts = self.case / 'scripts'
        scripts.mkdir(parents=True)
        (self.case / 'asymmetric_collision.cpp').touch()
        (self.case / 'inputs').mkdir()
        for name in ('asymmetric_collision.athinput', 'asymmetric_collision_amr.athinput',
                     'ac_amr_4gpu.athinput'):
            (self.case / 'inputs' / name).touch()
        executable = self.case / 'build/cuda-amg/src/athena'
        executable.parent.mkdir(parents=True)
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
        (scripts / 'run.slurm').write_text(
            'printf "%s\\n" "$NRANKS" "$RUN_DIR" "${UCX_NET_DEVICES:-unset}" '
            '"$INPUT_FILE" "$BUILD_DIR" "${ATHENA_REAL_EXE:-unset}" "$@"\n')
        shutil.copyfile(SCRIPTS / 'run_amg.slurm', scripts / 'run_amg.slurm')
        self.env = dict(os.environ)
        for key in ('NRANKS', 'ASYMMETRIC_CASE_DIR', 'RUN_DIR', 'MAX_NMB_PER_RANK',
                    'ATHENA_EXE', 'ATHENA_REAL_EXE', 'BUILD_DIR',
                    'ATHENA_WALLTIME', 'ATHENA_UCX_TLS',
                    'UCX_NET_DEVICES', 'ATHENA_UCX_NET_DEVICES', 'INPUT_FILE'):
            self.env.pop(key, None)
        self.env.update(SLURM_SUBMIT_DIR=str(self.case), SLURM_NTASKS='4',
                        SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='test')

    def launch(self, name='run_amg.slurm', args=(), cwd=None):
        # sbatch executes a copy from its spool directory, not scripts/.
        copied = self.root / 'slurm_script'
        shutil.copyfile(SCRIPTS / name, copied)
        return subprocess.run(['bash', str(copied), *args], cwd=cwd or self.root, env=self.env,
                              capture_output=True, text=True, timeout=5)

    def test_general_rank_capacity(self):
        for ranks, nodes, capacity in ((3, 1, 144), (4, 1, 108), (8, 2, 54)):
            with self.subTest(ranks=ranks):
                self.env.update(SLURM_NTASKS=str(ranks), SLURM_JOB_NUM_NODES=str(nodes))
                result = self.launch()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'mesh_refinement/max_nmb_per_rank={capacity}', result.stdout)
                self.assertIn('thermal_radiation/implicit_preconditioner=amr-amg',
                              result.stdout)
                self.assertEqual(result.stdout.splitlines()[0], str(ranks))
                self.assertEqual(result.stdout.splitlines()[2], 'eth0' if nodes > 1 else 'unset')

    def test_two_gpu_entry_point_after_slurm_copy(self):
        self.env['SLURM_NTASKS'] = '2'
        result = self.launch('run_amg_2gpu.slurm')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('mesh_refinement/max_nmb_per_rank=216', result.stdout)

    def test_explicit_capacity_restart_and_walltime(self):
        self.env['MAX_NMB_PER_RANK'] = '150'
        args = ('-r', '/checkpoint with spaces.rst', '-t', '00:10:00')
        result = self.launch(args=args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('mesh_refinement/max_nmb_per_rank=150', result.stdout)
        self.assertEqual(result.stdout.splitlines()[-4:], list(args))
        self.assertNotIn('23:45:00', result.stdout)

    def test_inconsistent_allocation_is_rejected(self):
        self.env['NRANKS'] = '8'
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('match the Slurm task allocation', result.stderr)

    def test_inter_node_interface_override(self):
        self.env.update(SLURM_JOB_NUM_NODES='2', ATHENA_UCX_NET_DEVICES='fabric0')
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[2], 'fabric0')

    def test_submission_from_inputs_directory(self):
        self.env['SLURM_SUBMIT_DIR'] = str(self.case / 'inputs')
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[3],
                         str(self.case / 'inputs/asymmetric_collision_amr.athinput'))

    def test_explicit_four_gpu_input_is_preserved(self):
        self.env['INPUT_FILE'] = str(self.case / 'inputs/ac_amr_4gpu.athinput')
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[3], self.env['INPUT_FILE'])
        self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)
        self.assertNotIn('thermal_radiation/implicit_preconditioner=', result.stdout)

    def test_restart_capacity_is_preserved_without_override(self):
        result = self.launch(args=('-r', '/new mesh checkpoint.rst'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)
        self.assertNotIn('thermal_radiation/implicit_preconditioner=', result.stdout)

    def test_command_line_input_capacity_is_preserved(self):
        result = self.launch(args=('-i', '/new mesh.athinput'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)
        self.assertNotIn('thermal_radiation/implicit_preconditioner=', result.stdout)

    def test_explicit_solver_override_is_forwarded(self):
        self.env['INPUT_FILE'] = str(self.case / 'inputs/ac_amr_4gpu.athinput')
        override = 'thermal_radiation/implicit_preconditioner=amr-amg'
        result = self.launch(args=(override,))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-1], override)
        self.assertEqual(result.stdout.count('thermal_radiation/implicit_preconditioner='), 1)

    def test_changed_mesh_does_not_use_legacy_capacity(self):
        for override in ('mesh/nx1=36', 'meshblock/nx1=36',
                         'mesh_refinement/num_levels=6'):
            with self.subTest(override=override):
                result = self.launch(args=(override,))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)

    def test_invalid_explicit_capacity_is_rejected(self):
        self.env['MAX_NMB_PER_RANK'] = '0'
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be a positive integer', result.stderr)

    def test_sc_four_gpu_defaults_after_slurm_copy(self):
        # --chdir selects the case even when sbatch was called from elsewhere.
        self.env['SLURM_SUBMIT_DIR'] = str(self.root / 'another submission directory')
        result = self.launch('slurms/run_sc_4gpu.sh', cwd=self.case)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], '4')
        self.assertEqual(lines[1], str(self.case / 'runs/sc-4gpu-test'))
        self.assertEqual(lines[3], str(self.case / 'inputs/ac_amr_4gpu.athinput'))
        self.assertEqual(lines[4], str(self.case / 'build/cuda-amg'))
        self.assertEqual(lines[5], str(self.case / 'build/cuda-amg/src/athena'))
        self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)
        self.assertNotIn('thermal_radiation/implicit_preconditioner=', result.stdout)

    def test_sc_four_gpu_missing_build_is_reported(self):
        (self.case / 'build/cuda-amg/src/athena').unlink()
        result = self.launch('slurms/run_sc_4gpu.sh', cwd=self.case)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing executable:', result.stderr)
        self.assertIn('sbatch scripts/make_amg.slurm', result.stderr)

    def test_sc_four_gpu_wrong_allocation_is_rejected(self):
        self.env['SLURM_NTASKS'] = '2'
        result = self.launch('slurms/run_sc_4gpu.sh', cwd=self.case)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('match the Slurm task allocation', result.stderr)

    def test_sc_four_gpu_restart_and_walltime_are_forwarded(self):
        self.env['ASYMMETRIC_CASE_DIR'] = str(self.case)
        args = ('-r', '/checkpoint with spaces.rst', '-t', '00:10:00')
        result = self.launch('slurms/run_sc_4gpu.sh', args=args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-4:], list(args))
        self.assertNotIn('mesh_refinement/max_nmb_per_rank=', result.stdout)
        self.assertNotIn('23:45:00', result.stdout)


if __name__ == '__main__':
    unittest.main()
