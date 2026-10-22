"""Batched cell AMG preserves each cell equation, tails, and restarted 3D fields."""
import os
import re

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr.test_nr_thermal_radiation_cell_amg import cell_amg_executable
from test_suite.nr.test_nr_thermal_radiation_amr_restart_mpicpu import execute


KAPPA = (100.0, 35.0, 1.0e7, 0.08, 70.0)


def input_deck(directory, groups, budget=1024):
    text = re.sub(r"\n<refined_region1>.*?(?=\n<)", "", utils.INPUT.read_text(), flags=re.S)
    # A dcycle parameter overrides dt even when output alignment is enabled.
    text = re.sub(r"(?m)^dcycle\s*=.*\n?", "", text)
    text += ("\n<thermal_radiation>\nimplicit_preconditioner = cell-amg\n"
             "implicit_amr_order = 1\nimplicit_group_batch_size = 4\n"
             "implicit_tolerance = 1e-12\nsolver_timers = true\n"
             f"implicit_amr_amg_max_mib = {budget}\nn_groups = {groups}\n")
    for group in range(groups):
        text += f"group_bound_{group} = {group}\nkappa_transport_{group} = {KAPPA[group]}\n"
    text += f"group_bound_{groups} = 100\n"
    text += ("\n<output1>\ndt = 0.0001\n"
             "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    path = directory / "batch.athinput"
    path.write_text(text)
    return path


def records(directory):
    # MPI can insert a complete rank report inside another rank's output line.
    return [dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in re.split(r"\n|(?=# radiation group batch:)",
                                 (directory / "run.log").read_text())
            if line.startswith("# radiation group batch:")]


@pytest.mark.parametrize("ranks,groups", ((1, 4), (3, 4), (4, 5)))
def test_cell_batch_dense_equations(tmp_path, cell_amg_executable, ranks, groups):
    deck = input_deck(tmp_path, groups)
    out = execute(tmp_path / "dense", ranks, ["-i", str(deck),
        "mesh/nx1=72", "meshblock/nx1=18", "problem/dr=1000000",
        "time/nlim=1", "time/tlim=0.0001"])
    initial, final = out[0], out[-1]
    centers = np.concatenate([bounds[0]+(np.arange(18)+0.5)*(bounds[1]-bounds[0])/18
                              for bounds in initial['mb_geometry']])
    order = np.argsort(centers)
    dx = np.diff(centers[order])[0]
    rho = np.asarray(initial['mb_data']['dens']).ravel()[order]
    dt = final['time']-initial['time']
    for group in range(groups):
        name = f'erad{group:02d}_d'
        rhs = np.asarray(initial['mb_data'][name]).ravel()[order]
        coefficient = 1000/(3*rho*KAPPA[group])
        matrix = np.eye(len(rhs))
        for i in range(len(rhs)):
            for j in ((i-1) % len(rhs), (i+1) % len(rhs)):
                rate = dt*0.5*(coefficient[i]+coefficient[j])/dx**2
                matrix[i, i] += rate
                matrix[i, j] -= rate
        actual = np.asarray(final['mb_data'][name]).ravel()[order]
        np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(actual.sum(), rhs.sum(), rtol=2e-10)
        assert actual.min() >= 0
    report = records(tmp_path / 'dense')
    assert len(report) == ranks
    assert all(int(r['size']) == 4 and int(r['setup_paired']) >= 1 for r in report)
    assert all(int(r['amg_requests']) > int(r['amg_batches']) for r in report)


def seed_3d(directory, deck, groups):
    out = execute(directory, 4, ['-i', str(deck), 'mesh/nx1=16', 'mesh/nx2=16',
        'mesh/nx3=16', 'meshblock/nx1=8', 'meshblock/nx2=8', 'meshblock/nx3=8',
        'mesh_refinement/max_nmb_per_rank=2', 'time/nlim=0'])
    checkpoint = sorted((directory / 'rst').glob('*.rst'))[0]
    payload = bytearray(checkpoint.read_bytes())
    shape = (out[0]['n_mbs'], 7+groups, 12, 12, 12)
    start = len(payload)-int(np.prod(shape))*8
    assert np.frombuffer(payload, dtype=np.uint64, count=1, offset=start-8)[0] == np.prod(shape[1:])*8
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    for block, bounds in enumerate(out[0]['mb_geometry']):
        h = (bounds[1]-bounds[0])/8
        axes = [bounds[2*a]+(np.arange(12)-1.5)*h for a in range(3)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        state[block, 0] = 1+0.25*np.cos(2*np.pi*y)
        for group in range(groups):
            state[block, 7+group] = (1+0.1*group+0.05*np.sin(2*np.pi*x+group)
                * np.cos(2*np.pi*y+group)*np.sin(2*np.pi*z+group))
    result = directory / 'seed.rst'
    result.write_bytes(payload)
    return result


def compare_groups(reference, candidate, groups):
    assert reference['cycle'] == candidate['cycle']
    left_topology, left = utils.ordered_fields(reference)
    right_topology, right = utils.ordered_fields(candidate)
    np.testing.assert_array_equal(left_topology, right_topology)
    for group in range(groups):
        name = f'erad{group:02d}_d'
        np.testing.assert_allclose(left[name], right[name], rtol=2e-9, atol=2e-11)
        assert np.isfinite(right[name]).all() and right[name].min() >= 0


@pytest.mark.parametrize('mode', ('packed', 'host'))
def test_cell_batch_repeated_3d_and_restart(tmp_path, cell_amg_executable, mode):
    deck = input_deck(tmp_path, 5)
    checkpoint = seed_3d(tmp_path / 'seed', deck, 5)
    flags = ['time/nlim=3', 'time/tlim=0.0003',
             f'thermal_radiation/implicit_halo_exchange={mode}']
    scalar = execute(tmp_path / 'scalar', 4, ['-r', str(checkpoint), *flags,
                     'thermal_radiation/implicit_group_batch_size=1'])
    batch = execute(tmp_path / 'batch', 4, ['-r', str(checkpoint), *flags])
    assert len(scalar) == len(batch) == 3 and batch[-1]['cycle'] == 3
    for before, after in zip(scalar, batch):
        compare_groups(before, after, 5)
        for group in range(5):
            np.testing.assert_allclose(utils.moments(after, f'erad{group:02d}_d')[0],
                utils.moments(batch[0], f'erad{group:02d}_d')[0], rtol=2e-10)
    report = records(tmp_path / 'batch')
    assert len(report) == 12 and all(int(r['size']) == 4 for r in report)
    assert all(int(r['scalar_retries']) == 0 for r in report)
    checkpoint = sorted((tmp_path / 'batch/rst').glob('*.rst'))[0]
    resumed = execute(tmp_path / 'resumed', 4, ['-r', str(checkpoint), *flags])
    compare_groups(batch[-1], resumed[-1], 5)


@pytest.mark.skipif('ATHENA_GROUP_BATCH_TEST_TOTAL_MIB' not in os.environ,
                   reason='requires known GPU capacity')
def test_cell_batch_memory_fallback(tmp_path, cell_amg_executable):
    budget = 3*int(os.environ['ATHENA_GROUP_BATCH_TEST_TOTAL_MIB'])//5
    deck = input_deck(tmp_path, 4, budget)
    flags = ['-i', str(deck), 'mesh/nx1=32', 'meshblock/nx1=8', 'time/nlim=2']
    scalar = execute(tmp_path / 'scalar', 4,
                     [*flags, 'thermal_radiation/implicit_group_batch_size=1'])
    batch = execute(tmp_path / 'batch', 4, flags)
    for before, after in zip(scalar, batch):
        compare_groups(before, after, 4)
    report = records(tmp_path / 'batch')
    assert report and all(r['size'] == '1' and r['reason'] == 'workspace_budget' for r in report)
