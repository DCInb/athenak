"""Independent equations, unequal group progress, and multi-peer MPI batching."""
import os
import re

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr import test_nr_thermal_radiation_amg_cpu as serial
from test_suite.nr import test_nr_thermal_radiation_amg_mpicpu as distributed
from test_suite.nr.test_nr_thermal_radiation_amg_cpu import amg_executable  # noqa: F401


@pytest.fixture
def batch_input(tmp_path, monkeypatch):
    deck = tmp_path / "batch.athinput"
    deck.write_text(utils.INPUT.read_text()+
                    "\n<thermal_radiation>\nimplicit_group_batch_size = 2\n")
    monkeypatch.setattr(utils, "INPUT", deck)
    return deck


@pytest.mark.parametrize("order,inner", ((1, 0), (2, 0), (2, 20)))
def test_batched_stiff_equations(tmp_path, amg_executable, batch_input, order, inner):
    # Independently assembled composite finite-volume residuals for BOTH groups,
    # with distinct opacities, plus a million-fold jump across rank ownership.
    distributed.test_amg_stiff_true_equation(tmp_path, amg_executable, order, inner)


@pytest.mark.parametrize("ranks", (1, 2))
def test_batched_dense_reference(tmp_path, amg_executable, batch_input, ranks):
    serial.test_amg_periodic_dense_reference(tmp_path, amg_executable, ranks)


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
@pytest.mark.parametrize("ranks", (2, 3, 4))
def test_batched_regrid(tmp_path, amg_executable, batch_input, monkeypatch, adaptive, ranks):
    flags = (*distributed.BLOCK18, "mesh_refinement/max_nmb_per_rank=32",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = utils.run_case(tmp_path / "scalar", 2, "amr-amg", ranks=ranks,
        adaptive=adaptive, extra_flags=(*flags, "thermal_radiation/implicit_group_batch_size=1"))
    candidate = utils.run_case(tmp_path / "batch", 2, "amr-amg", ranks=ranks,
        adaptive=adaptive, extra_flags=flags)
    utils.assert_transport(candidate, adaptive)
    utils.assert_same_solution(reference, candidate)


@pytest.mark.parametrize("mode", ("packed", "host"))
@pytest.mark.parametrize("groups", (2, 4))
def test_batched_unequal_progress(tmp_path, amg_executable, batch_input, mode, groups):
    # The nearly static group retires early; the other must continue with the
    # same frozen preconditioner and unmodified convergence criterion.
    if groups == 4:
        with batch_input.open('a') as deck:
            deck.write('\nn_groups = 4\nimplicit_group_batch_size = 4\n'
                       'group_bound_2 = 2\ngroup_bound_3 = 3\ngroup_bound_4 = 100\n'
                       'kappa_transport_2 = 10000000\nkappa_transport_3 = 0.08\n')
    flags = (*distributed.BLOCK18, "time/nlim=2", "time/tlim=0.0002",
             "thermal_radiation/kappa_transport_0=10000000",
             "thermal_radiation/kappa_transport_1=0.035",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12",
             f"thermal_radiation/implicit_halo_exchange={mode}")
    reference = utils.run_case(tmp_path / "scalar", 2, "amr-amg", ranks=2,
        extra_flags=(*flags, "thermal_radiation/implicit_group_batch_size=1"))
    candidate = utils.run_case(tmp_path / "batch", 2, "amr-amg", ranks=2, extra_flags=flags)
    utils.assert_same_solution(reference, candidate)
    if groups == 4:
        for before, after in zip(reference, candidate):
            a, b = utils.ordered_fields(before)[1], utils.ordered_fields(after)[1]
            for group in ('erad02_d', 'erad03_d'):
                np.testing.assert_allclose(a[group], b[group], rtol=2e-9, atol=2e-11)
    log = (tmp_path / "batch/run.log").read_text()
    records = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
               if line.startswith("# radiation group batch:")]
    assert len(records) == 4
    assert all(int(r['setup_paired']) == 1 for r in records)
    assert any(int(r['amg_requests']) > int(r['amg_batches']) for r in records)
    assert any(int(r['halo_requests']) > int(r['halo_batches']) for r in records)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_batched_boundaries(tmp_path, amg_executable, batch_input, boundary):
    distributed.test_amg_physical_boundary_reference(tmp_path, amg_executable, boundary)


@pytest.mark.parametrize('groups,batch_size', ((3, 2), (3, 4), (4, 4), (5, 4)))
def test_batched_group_count(tmp_path, amg_executable, batch_input, groups, batch_size):
    with batch_input.open("a") as deck:
        deck.write(f'n_groups = {groups}\nimplicit_group_batch_size = {batch_size}\n')
        for g in range(2, groups):
            deck.write(f'group_bound_{g} = {g}\nkappa_transport_{g} = {2.5*g}\n')
        deck.write(f'group_bound_{groups} = 100\n')
    flags = (*distributed.BLOCK18, "time/nlim=2", "time/tlim=0.0002",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    ref = utils.run_case(tmp_path / "scalar", 2, "amr-amg", ranks=2,
        extra_flags=(*flags, "thermal_radiation/implicit_group_batch_size=1"))
    out = utils.run_case(tmp_path / "batch", 2, "amr-amg", ranks=2, extra_flags=flags)
    assert len(ref) == len(out)
    for before, after in zip(ref, out):
        assert before['cycle'] == after['cycle'] and before['time'] == after['time']
        topology, fields = utils.ordered_fields(before)
        actual_topology, actual_fields = utils.ordered_fields(after)
        np.testing.assert_array_equal(topology, actual_topology)
        assert 'erad02_d' in fields
        for group in (f'erad{g:02d}_d' for g in range(groups)):
            assert np.all(np.isfinite(actual_fields[group])) and np.all(actual_fields[group] >= 0)
            np.testing.assert_allclose(fields[group], actual_fields[group], rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("ranks", (2, 3, 4))
def test_batched_adapter_oracle(tmp_path, monkeypatch, ranks):
    distributed.test_amg_collective_adapter_failure(tmp_path, monkeypatch, 'batch-oracle', ranks)


@pytest.mark.parametrize("mode", ("packed", "host"))
@pytest.mark.parametrize("dimensions,ranks", ((1, 3), (1, 4), (2, 3), (2, 4), (3, 4)))
def test_batched_multiple_peers(tmp_path, amg_executable, batch_input, mode, dimensions, ranks):
    # Five independent equations exercise a four-lane batch and a scalar tail.
    # The 1D layout includes ranks without remote reflux faces; 2D/3D introduce
    # multiple peers, unequal block ownership, and directed coarse/fine traffic.
    with batch_input.open('a') as deck:
        deck.write('\nn_groups = 5\nimplicit_group_batch_size = 4\n'
                   'group_bound_2 = 2\ngroup_bound_3 = 3\ngroup_bound_4 = 4\n'
                   'group_bound_5 = 100\nkappa_transport_2 = 80\n'
                   'kappa_transport_3 = 0.08\nkappa_transport_4 = 70\n')
    mesh = (('mesh/nx1=144', 'meshblock/nx1=18') if dimensions == 1 else
            (*distributed.BLOCK18, *(['mesh/nx3=36', 'meshblock/nx3=18',
                                    'refined_region1/x3max=0.5'] if dimensions == 3 else [])))
    flags = (*mesh, 'time/nlim=2', 'time/tlim=0.0002',
             'thermal_radiation/kappa_transport_0=10000000',
             'thermal_radiation/kappa_transport_1=0.035',
             'thermal_radiation/implicit_amr_inner_iterations=20',
             'thermal_radiation/implicit_tolerance=1e-12',
             f'thermal_radiation/implicit_halo_exchange={mode}')
    ref = utils.run_case(tmp_path / 'scalar', dimensions, 'amr-amg', ranks=1,
        extra_flags=(*flags, 'thermal_radiation/implicit_group_batch_size=1'))
    out = utils.run_case(tmp_path / 'batch', dimensions, 'amr-amg', ranks=ranks,
        extra_flags=flags)
    assert len(ref) == len(out)
    for before, after in zip(ref, out):
        assert before['cycle'] == after['cycle'] and before['time'] == after['time']
        topology, fields = utils.ordered_fields(before)
        actual_topology, actual = utils.ordered_fields(after)
        np.testing.assert_array_equal(topology, actual_topology)
        for name in (f'erad{g:02d}_d' for g in range(5)):
            assert np.all(np.isfinite(actual[name])) and np.all(actual[name] >= 0)
            np.testing.assert_allclose(actual[name], fields[name], rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(utils.moments(after, name)[0],
                                       utils.moments(out[0], name)[0], rtol=2e-10, atol=2e-12)
    log = (tmp_path / 'batch/run.log').read_text()
    records = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in log.splitlines()
               if line.startswith('# radiation group batch:')]
    assert len(records) == 2*ranks
    assert {int(r['rank']) for r in records} == set(range(ranks))
    assert all(int(r['size']) == 4 and int(r['setup_paired']) == 1 for r in records)
    assert all(int(r['halo_requests']) > int(r['halo_batches']) for r in records)
    # All ranks must schedule the same cohorts, even with an empty reflux plan.
    for cycle in {r['cycle'] for r in records}:
        for key in ('setup_requests', 'setup_batches', 'amg_requests', 'amg_batches',
                    'halo_requests', 'halo_batches', 'reduction_requests', 'reduction_batches'):
            assert len({r[key] for r in records if r['cycle'] == cycle}) == 1
    assert 'AMR fallback' not in log


@pytest.mark.parametrize("mode", ("packed", "host"))
def test_batched_four_rank_dense_reference(tmp_path, amg_executable, batch_input, mode):
    out = utils.run_case(tmp_path / 'dense', 1, 'amr-amg', ranks=4, extra_flags=(
        'mesh/nx1=144', 'meshblock/nx1=18', 'thermal_radiation/implicit_amr_order=1',
        'thermal_radiation/implicit_tolerance=1e-12',
        f'thermal_radiation/implicit_halo_exchange={mode}'))
    utils.assert_1d_reference(out)


@pytest.mark.skipif('ATHENA_GROUP_BATCH_TEST_TOTAL_MIB' not in os.environ,
                    reason='requires a CUDA run with known device capacity')
@pytest.mark.parametrize('batch_size', (2, 4))
@pytest.mark.parametrize('ranks', (2, 4))
def test_batched_workspace_fallback(tmp_path, amg_executable, batch_input, batch_size, ranks):
    # Reserve more than half the physical GPU for each AMG workspace. One fits,
    # while two cannot: exercise the real collective preflight without allocating
    # that memory or changing the physical equations.
    budget = 3*int(os.environ['ATHENA_GROUP_BATCH_TEST_TOTAL_MIB'])//5
    with batch_input.open('a') as deck:
        deck.write(f'\nimplicit_amr_amg_max_mib = {budget}\n')
        if batch_size == 4:
            deck.write('n_groups = 4\nimplicit_group_batch_size = 4\n'
                       'group_bound_2 = 2\ngroup_bound_3 = 3\ngroup_bound_4 = 100\n'
                       'kappa_transport_2 = 80\nkappa_transport_3 = 70\n')
    flags = (*distributed.BLOCK18, 'thermal_radiation/implicit_tolerance=1e-12')
    reference = utils.run_case(tmp_path / 'scalar', 2, 'amr-amg', ranks=ranks,
        extra_flags=(*flags, 'thermal_radiation/implicit_group_batch_size=1'))
    candidate = utils.run_case(tmp_path / 'batch', 2, 'amr-amg', ranks=ranks,
        extra_flags=flags)
    utils.assert_same_solution(reference, candidate)
    for before, after in zip(reference, candidate):
        a, b = utils.ordered_fields(before)[1], utils.ordered_fields(after)[1]
        for g in range(batch_size):
            name = f'erad{g:02d}_d'
            np.testing.assert_allclose(a[name], b[name], rtol=2e-9, atol=2e-11)
    records = [line for line in (tmp_path / 'batch/run.log').read_text().splitlines()
               if line.startswith('# radiation group batch:')]
    assert len(records) >= 2
    assert all('size=1 reason=workspace_budget' in line for line in records)


@pytest.mark.parametrize("batch_ranks", (2, 4))
def test_batched_positivity_retry(tmp_path, amg_executable, batch_input, batch_ranks):
    from test_suite.nr.test_nr_thermal_radiation_amr_cholmod_mpicpu import _run
    from test_suite.nr.test_nr_thermal_radiation_amr_aggregate_mpicpu import aggregate_leaf_geometry
    from test_suite.nr.test_nr_thermal_radiation_amr_preconditioner_mpicpu import monotone_diffusion
    dt, ghosts, cells = 1e-5, 2, 18
    with batch_input.open('a') as deck:
        deck.write('\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n')
    flags = (*distributed.BLOCK18, 'time/nlim=0', f'time/initial_dt={dt}',
        f'time/tlim={dt}', 'time/align_outputs=false', 'thermal_radiation/c_light=3',
        'thermal_radiation/kappa_transport_0=1', 'thermal_radiation/kappa_transport_1=3',
        'thermal_radiation/implicit_preconditioner=amr-amg',
        'thermal_radiation/implicit_amr_inner_iterations=20',
        'thermal_radiation/implicit_tolerance=1e-12')
    initial = _run(tmp_path / 'source', ['-i', str(batch_input), *flags])
    payload = bytearray(sorted((tmp_path / 'source/rst').glob('*.rst'))[-1].read_bytes())
    extent = cells+2*ghosts
    shape = (initial['n_mbs'], 9, 1, extent, extent)
    per_block = int(np.prod(shape[1:])*8)
    start = len(payload)-shape[0]*per_block
    assert np.frombuffer(payload, dtype=np.uint64, count=1, offset=start-8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    rhs = []
    for block, bounds in enumerate(initial['mb_geometry']):
        width = (bounds[1]-bounds[0])/cells
        x = bounds[0]+(np.arange(extent)-ghosts+0.5)*width
        y = bounds[2]+(np.arange(extent)-ghosts+0.5)*width
        yy, xx = np.meshgrid(y, x, indexing='ij')
        values = np.full((extent, extent), 1e-12)
        values[np.isclose(xx, -0.5/36) & np.isclose(yy, 0.5/36)] = 1
        state[block, 7, 0] = values
        state[block, 8, 0] = 0.25
        rhs.extend(values[ghosts:-ghosts, ghosts:-ghosts].ravel())
    injected = tmp_path / 'impulse.rst'
    injected.write_bytes(payload)
    rhs = np.asarray(rhs)
    assert np.count_nonzero(rhs == 1) == 1
    actuals = []
    for name, order, ranks, batch in [('scalar', 1, 1, 1), ('batch', 2, batch_ranks, 2)]:
        output = _run(tmp_path / name, ['-r', str(injected), 'time/nlim=1',
            f'thermal_radiation/implicit_amr_order={order}',
            f'thermal_radiation/implicit_group_batch_size={batch}'], ranks)
        owner, widths = aggregate_leaf_geometry(output)
        actual = np.asarray(output['mb_data']['erad00_d']).ravel()
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
        action = monotone_diffusion(actual, owner, widths, np.ones_like(actual))
        np.testing.assert_allclose(actual+dt*action, rhs, rtol=0, atol=3e-10)
        np.testing.assert_allclose(np.dot(widths**2, actual), np.dot(widths**2, rhs),
                                   rtol=0, atol=2e-12)
        np.testing.assert_allclose(output['mb_data']['erad01_d'], 0.25, rtol=0, atol=1e-13)
        actuals.append(utils.ordered_fields(output)[1]['erad00_d'])
    np.testing.assert_allclose(actuals[0], actuals[1], rtol=0, atol=3e-10)
    log = (tmp_path / 'batch/run.log').read_text()
    assert 'AMR fallback: group=0 reason=negative_energy' in log
    assert 'scalar_retries=1' in log
