"""GPU CPU binding must use local ranks and this node's Slurm allocation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'map_gpu_rank', Path(__file__).resolve().parents[1] / 'scripts/map_gpu_rank.py')
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)


class AllocationTests(unittest.TestCase):
    @patch.object(mapper.subprocess, 'check_output')
    def test_node_specific_allocation(self, command):
        command.side_effect = ['node-a\n', 'node-b\n']
        job = ('Nodes=node-a CPU_IDs=0-3 Mem=100\n'
               'Nodes=node-b CPU_IDs=64-67 Mem=100\n')
        self.assertEqual(mapper.allocated_cpus(job, 'node-b.example'), set(range(64, 68)))

    @patch.object(mapper.subprocess, 'check_output', return_value='node-a\nnode-b\n')
    def test_compressed_node_list(self, command):
        self.assertEqual(mapper.allocated_cpus(
            'Nodes=node-[a-b] CPU_IDs=1,3,5-6 Mem=100', 'node-b'), {1, 3, 5, 6})

    @patch.object(mapper.subprocess, 'check_output', return_value='node-a\n')
    def test_missing_node_is_rejected(self, command):
        with self.assertRaisesRegex(RuntimeError, 'No Slurm CPU allocation'):
            mapper.allocated_cpus('Nodes=node-a CPU_IDs=0-3 Mem=100', 'node-b')

    def test_global_rank_does_not_index_local_devices(self):
        chosen = dict(cpu=65, uuid='gpu-b', pci='0000:81:00.0')
        with patch.dict(mapper.os.environ, {'OMPI_COMM_WORLD_RANK': '5',
                'OMPI_COMM_WORLD_LOCAL_RANK': '1', 'ATHENA_REAL_EXE': __file__}), \
             patch.object(mapper.sys, 'argv', ['map_gpu_rank.py', '-n']), \
             patch.object(mapper, 'gpu_cpu_plan', return_value={1: chosen}), \
             patch.object(mapper.os, 'sched_setaffinity') as affinity, \
             patch.object(mapper.os, 'execv') as execute, patch('builtins.print'):
            mapper.main()
        affinity.assert_called_once_with(0, {65})
        self.assertEqual(execute.call_args.args[1][1:], ['-n', '--kokkos-map-device-id-by=mpi_rank'])


if __name__ == '__main__':
    unittest.main()
