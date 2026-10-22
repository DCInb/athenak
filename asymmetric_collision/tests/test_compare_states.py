"""Ignore unused restart-header bytes while retaining physical metadata gates."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'compare_states', Path(__file__).resolve().parents[1] / 'scripts/compare_states.py')
compare_states = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_states)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(blocks=1, root_level=0, geometry=(0, 1)*3,
            mesh_indices=tuple(range(19)), block_indices=tuple(range(19)), time=1,
            cycle=1, dt=0.1, locations=np.zeros((1, 4)), ages=np.zeros(1),
            labels=['density', 'mx', 'my', 'mz', 'energy', 'ion', 'electron'],
            path='reference', volumes=np.ones(1), scalars=0, cells=1,
            fields=np.ones((1, 7, 1, 1, 1)), active=(slice(None),)*4)

    def test_unused_root_entries_do_not_change_physical_state(self):
        candidate = dict(self.reference, mesh_indices=tuple(range(10))+(123456,)*9)
        result = compare_states.compare(self.reference, candidate, 5e-7)
        self.assertTrue(result['passed'])
        self.assertTrue(result['bitwise_identical'])

    def test_active_root_indices_are_checked(self):
        values = list(self.reference['mesh_indices'])
        values[1] += 1
        result = compare_states.compare(self.reference, dict(self.reference,
            mesh_indices=tuple(values)), 5e-7)
        self.assertFalse(result['passed'])
        self.assertIn('mesh_indices', result['metadata_mismatches'])

    def test_block_coarse_indices_are_checked(self):
        values = list(self.reference['block_indices'])
        values[10] += 1
        result = compare_states.compare(self.reference, dict(self.reference,
            block_indices=tuple(values)), 5e-7)
        self.assertFalse(result['passed'])
        self.assertIn('block_indices', result['metadata_mismatches'])

    def test_time_roundoff_requires_explicit_allowance(self):
        candidate = dict(self.reference, time=1.0+1e-14)
        strict = compare_states.compare(self.reference, candidate, 5e-7)
        self.assertFalse(strict['passed'])
        self.assertIn('time', strict['metadata_mismatches'])
        result = compare_states.compare(self.reference, candidate, 5e-7, time_tolerance=1e-12)
        self.assertTrue(result['passed'])
        self.assertFalse(result['time_exactly_equal'])
        self.assertEqual(result['candidate_time'], candidate['time'])

    def test_time_allowance_rejects_different_physical_time(self):
        result = compare_states.compare(self.reference, dict(self.reference, time=1.0+1e-8),
                                        5e-7, time_tolerance=1e-12)
        self.assertFalse(result['passed'])
        self.assertIn('time', result['metadata_mismatches'])

    def test_time_allowance_does_not_weaken_field_gate(self):
        candidate = dict(self.reference, time=1.0+1e-14, fields=self.reference['fields'].copy())
        candidate['fields'][0, -1, 0, 0, 0] += 1e-4
        result = compare_states.compare(self.reference, candidate, 5e-7, time_tolerance=1e-12)
        self.assertEqual(result['metadata_mismatches'], [])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
