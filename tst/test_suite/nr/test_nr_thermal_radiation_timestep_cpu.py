"""Focused stability tests for explicit and implicit multigroup FLD transport."""

import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pytest

import test_suite.testutils as testutils


diffusion_input = "../../../inputs/hydro/mgfld_diffusion.athinput"
table_input = "../../../inputs/hydro/mgfld_opacity_table_diffusion.athinput"
relax_input = "../../../inputs/hydro/two_temperature_mgfld.athinput"
opacity_table = "../../../inputs/hydro/two_temperature_opacity_table.dat"


def run_case(input_file, flags, timeout=60.0):
    """Run AthenaK directly so the cycle diagnostics (including dt) are available."""
    command = ["./athena", "-i", input_file, *flags]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"AthenaK failed with return code {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    timesteps = [
        float(value) for value in re.findall(
            r"cycle=\d+\s+time=[^\s]+\s+dt=([^\s]+)", result.stdout)
    ]
    if not timesteps:
        pytest.fail(f"No timestep diagnostics found in output:\n{result.stdout}")
    return timesteps


def transport_flags(basename, resolution=64, opacity=1.0e-8, nlim=0):
    block = min(32, resolution//2)
    return [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}",
        f"meshblock/nx1={block}",
        f"time/nlim={nlim}",
        "time/tlim=1.0",
        "problem/pl=1.0e-8",
        "problem/pr=1.0e-8",
        f"thermal_radiation/kappa_transport_0={opacity}",
        f"thermal_radiation/kappa_transport_1={opacity}",
        "output1/dt=1.0",
    ]


def read_tab(path):
    """Read the small ASCII slice outputs without adding a NumPy dependency here."""
    names = None
    columns = {}
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("# gid"):
                names = line[1:].split()
                columns = {name: [] for name in names}
            elif names is not None and line.strip() and not line.startswith("#"):
                values = line.split()
                for name, value in zip(names, values):
                    columns[name].append(float(value))
    if not columns:
        raise AssertionError(f"No tabular data found in {path}")
    return columns


def input_variant(source, directory, name, updates):
    """Create an input variant, replacing existing keys and adding new block keys."""
    text = Path(source).read_text()
    for block, parameters in updates.items():
        match = re.search(
            rf"(?ms)^<{re.escape(block)}>\s*$.*?(?=^<|\Z)", text)
        if match is None:
            raise AssertionError(f"Input block <{block}> not found in {source}")
        block_text = match.group(0)
        for key, value in parameters.items():
            pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
            replacement = f"{key} = {value}"
            if re.search(pattern, block_text):
                block_text = re.sub(pattern, replacement, block_text, count=1)
            else:
                block_text = block_text.rstrip() + "\n" + replacement + "\n\n"
        text = text[:match.start()] + block_text + text[match.end():]
    path = Path(directory)/f"{name}.athinput"
    path.write_text(text)
    return str(path)


def assert_slice_conserved_and_positive(basename):
    initial = read_tab(f"tab/{basename}.hydro_3t.00000.tab")
    final = read_tab(f"tab/{basename}.hydro_3t.00001.tab")
    for group in ("erad00", "erad01"):
        assert min(final[group]) >= 0.0
    assert math.isclose(
        sum(final["erad"]), sum(initial["erad"]), rel_tol=2.0e-10,
        abs_tol=2.0e-10)


def solve_dense(matrix, values):
    """Small dependency-free dense solve with partial pivoting."""
    count = len(values)
    rhs = list(values)
    for column in range(count):
        pivot = max(range(column, count), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        diagonal = matrix[column][column]
        for row in range(column + 1, count):
            factor = matrix[row][column]/diagonal
            if factor == 0.0:
                continue
            matrix[row][column] = 0.0
            for entry in range(column + 1, count):
                matrix[row][entry] -= factor*matrix[column][entry]
            rhs[row] -= factor*rhs[column]

    solution = [0.0 for _ in range(count)]
    for row in range(count - 1, -1, -1):
        remainder = sum(
            matrix[row][entry]*solution[entry]
            for entry in range(row + 1, count))
        solution[row] = (rhs[row] - remainder)/matrix[row][row]
    return solution


def periodic_backward_euler(values, coefficient):
    """Dense reference for (I-coefficient*periodic_laplacian) x = values."""
    count = len(values)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    for row in range(count):
        matrix[row][row] = 1.0 + 2.0*coefficient
        matrix[row][(row - 1) % count] -= coefficient
        matrix[row][(row + 1) % count] -= coefficient
    return solve_dense(matrix, values)


def harmonic_frozen_backward_euler(values, opacity, light_speed, timestep):
    """Reference the implicit solver's old-state harmonic FLD matrix exactly."""
    count = len(values)
    dx = 1.0/count
    energy_floor = 1.0e-30
    alpha = 1.0
    sigma = max(opacity, 1.0e-30)
    transport_coefficients = []
    for cell, energy in enumerate(values):
        gradient = abs(values[(cell + 1) % count]
                       - values[(cell - 1) % count])/(2.0*dx)
        effective_energy = max(energy, energy_floor)
        limiter_argument = gradient/(sigma*effective_energy*alpha)
        diffusion_coefficient = 1.0/(3.0 + limiter_argument)/sigma
        roundoff_gradient = (64.0*sys.float_info.epsilon
                             * max(abs(energy), energy_floor)/dx)
        if gradient <= roundoff_gradient:
            diffusion_coefficient = min(
                diffusion_coefficient, 0.5*alpha*dx)
        transport_coefficients.append(light_speed*diffusion_coefficient)

    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    for cell in range(count):
        right = 0.5*(transport_coefficients[cell]
                     + transport_coefficients[(cell + 1) % count])
        left = 0.5*(transport_coefficients[cell]
                    + transport_coefficients[(cell - 1) % count])
        factor = timestep/(dx*dx)
        matrix[cell][cell] = 1.0 + factor*(left + right)
        matrix[cell][(cell - 1) % count] -= factor*left
        matrix[cell][(cell + 1) % count] -= factor*right
    return solve_dense(matrix, values), transport_coefficients


def harmonic_vacuum_backward_euler(values, opacity, light_speed, timestep):
    """Reference the centered implicit harmonic matrix with zero-radiation ghosts."""
    count = len(values)
    dx = 1.0/count
    energy_floor = 1.0e-30
    alpha = 1.0
    sigma = max(opacity, 1.0e-30)
    transport_coefficients = []
    for cell, energy in enumerate(values):
        left = values[cell - 1] if cell > 0 else 0.0
        right = values[cell + 1] if cell + 1 < count else 0.0
        gradient = abs(right - left)/(2.0*dx)
        effective_energy = max(energy, energy_floor)
        limiter_argument = gradient/(sigma*effective_energy*alpha)
        diffusion_coefficient = 1.0/(3.0 + limiter_argument)/sigma
        roundoff_gradient = (64.0*sys.float_info.epsilon
                             * max(abs(energy), energy_floor)/dx)
        if gradient <= roundoff_gradient:
            diffusion_coefficient = min(
                diffusion_coefficient, 0.5*alpha*dx)
        transport_coefficients.append(light_speed*diffusion_coefficient)

    vacuum_cap = 0.5*light_speed*alpha*dx
    face_coefficients = [vacuum_cap]
    face_coefficients.extend(
        0.5*(transport_coefficients[cell]
             + transport_coefficients[cell + 1])
        for cell in range(count - 1))
    face_coefficients.append(vacuum_cap)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    factor = timestep/(dx*dx)
    for cell in range(count):
        left = face_coefficients[cell]
        right = face_coefficients[cell + 1]
        matrix[cell][cell] = 1.0 + factor*(left + right)
        if cell > 0:
            matrix[cell][cell - 1] -= factor*left
        if cell + 1 < count:
            matrix[cell][cell + 1] -= factor*right
    return solve_dense(matrix, values), transport_coefficients, vacuum_cap


def test_run():
    variants_context = tempfile.TemporaryDirectory(prefix="athenak-fld-dt-")
    variants = variants_context.name
    try:
        # In the diffusion limit the face Jacobian must reproduce the usual explicit
        # bound: cfl*[2*c*D/dx^2]^-1, with D=1/(3*rho*kappa).
        diffusion = transport_flags("fld_dt_diffusion", opacity=100.0)
        diffusion.extend([
            "thermal_radiation/initial_radiation_temperature_right=0.999999",
        ])
        dt_diffusion = run_case(diffusion_input, diffusion)[0]
        expected_diffusion = 0.4*1.5*100.0/64.0**2
        assert dt_diffusion == pytest.approx(expected_diffusion, rel=2.0e-6)

        # Directional face maxima must remain independent before their rates are
        # summed.  With D=1/(3*kappa), the anisotropic 3-D mesh gives distinct
        # x1:x2:x3 contributions of 16^2:8^2:4^2.
        anisotropic = transport_flags(
            "fld_dt_anisotropic_3d", resolution=16, opacity=100.0)
        anisotropic.extend([
            "mesh/nx2=8", "mesh/nx3=4",
            "meshblock/nx2=8", "meshblock/nx3=4",
            "thermal_radiation/initial_profile=uniform",
            "thermal_radiation/flux_limiter=none",
        ])
        expected_anisotropic = 0.4*1.5*100.0/(16.0**2 + 8.0**2 + 4.0**2)
        assert run_case(diffusion_input, anisotropic)[0] == pytest.approx(
            expected_anisotropic, rel=2.0e-6)

        # In the optically thin limit dt scales as dx/c_hat, not as
        # rho*kappa*dx^2/c_hat.  It is invariant to another four decades in opacity.
        streaming_dt = {}
        for resolution in (32, 64, 128):
            flags = transport_flags(
                f"fld_dt_stream_{resolution}", resolution=resolution)
            streaming_dt[resolution] = run_case(diffusion_input, flags)[0]
            assert streaming_dt[resolution] == pytest.approx(
                0.4/resolution, rel=2.0e-6)
        assert streaming_dt[32]/streaming_dt[64] == pytest.approx(2.0)
        assert streaming_dt[64]/streaming_dt[128] == pytest.approx(2.0)
        thinner = transport_flags("fld_dt_thinner", opacity=1.0e-12)
        assert run_case(diffusion_input, thinner)[0] == pytest.approx(
            streaming_dt[64], rel=2.0e-6)

        # FLASH disables its diffusion timestep because transport is backward implicit.
        # With matter coupling disabled and source_cfl=0, changing physical c by eight
        # decades therefore cannot change the AthenaK macro timestep.
        implicit_dt = []
        for light_speed in (1.0, 1.0e8):
            flags = transport_flags(
                f"fld_implicit_dt_{light_speed:g}", opacity=100.0)
            flags.extend([
                "thermal_radiation/transport_integrator=implicit",
                "thermal_radiation/source_cfl=0",
                f"thermal_radiation/c_light={light_speed}",
            ])
            implicit_dt.append(run_case(diffusion_input, flags)[0])
        assert implicit_dt == pytest.approx([1.0, 1.0], rel=0.0, abs=1.0e-14)

        invalid_tolerance_input = input_variant(
            diffusion_input, variants, "invalid_implicit_tolerance",
            {"thermal_radiation": {
                "transport_integrator": "implicit",
                "implicit_tolerance": "1.0e-20",
            }})
        invalid_tolerance = subprocess.run(
            ["./athena", "-i", invalid_tolerance_input],
            text=True, capture_output=True, timeout=60.0, check=False)
        assert invalid_tolerance.returncode != 0
        assert "Implicit radiation tolerance must be finite" in (
            invalid_tolerance.stdout + invalid_tolerance.stderr)

        # The transport solve still uses physical c.  Compare a fixed macro step at two
        # light speeds against the exact periodic backward-Euler finite-volume stencil.
        implicit_results = {}
        implicit_step = 1.0e-3
        implicit_opacity = 100.0
        implicit_resolution = 64
        for light_speed in (1.0, 2.0):
            basename = f"fld_implicit_c{light_speed:g}"
            flags = transport_flags(
                basename, resolution=implicit_resolution,
                opacity=implicit_opacity, nlim=1)
            flags.extend([
                f"time/tlim={implicit_step}",
                f"output1/dt={implicit_step}",
                "thermal_radiation/transport_integrator=implicit",
                "thermal_radiation/source_cfl=0",
                "thermal_radiation/flux_limiter=none",
                f"thermal_radiation/c_light={light_speed}",
            ])
            assert run_case(diffusion_input, flags)[0] == pytest.approx(implicit_step)
            initial = read_tab(f"tab/{basename}.hydro_3t.00000.tab")
            final = read_tab(f"tab/{basename}.hydro_3t.00001.tab")
            stencil_coefficient = (implicit_step*light_speed
                                   * implicit_resolution**2
                                   / (3.0*implicit_opacity))
            for group in ("erad00", "erad01"):
                reference = periodic_backward_euler(
                    initial[group], stencil_coefficient)
                assert final[group] == pytest.approx(
                    reference, rel=3.0e-9, abs=3.0e-11)
                assert min(final[group]) >= 0.0
                assert sum(final[group]) == pytest.approx(
                    sum(initial[group]), rel=3.0e-11, abs=3.0e-11)
            implicit_results[light_speed] = final["erad"]

        mean_radiation = sum(implicit_results[1.0])/implicit_resolution
        variance_c1 = sum(
            (value - mean_radiation)**2 for value in implicit_results[1.0])
        variance_c2 = sum(
            (value - mean_radiation)**2 for value in implicit_results[2.0])
        assert variance_c2 < variance_c1

        # A resolved optically thin gradient must retain harmonic FLD's
        # D~E/|grad(E)| coefficient.  Only roundoff-flat cells receive the grid-scale
        # regularization.  Compare two resolutions with the exact frozen-coefficient
        # variable-matrix solve; an unconditional D<=dx/2 cap fails this comparison.
        limited_opacity = 1.0e-8
        limited_step = 5.0e-4
        for resolution in (32, 64):
            basename = f"fld_implicit_harmonic_{resolution}"
            flags = transport_flags(
                basename, resolution=resolution,
                opacity=limited_opacity, nlim=1)
            flags.extend([
                f"time/tlim={limited_step}",
                f"output1/dt={limited_step}",
                "thermal_radiation/transport_integrator=implicit",
                "thermal_radiation/source_cfl=0",
                "thermal_radiation/flux_limiter=harmonic",
                "thermal_radiation/c_light=1.0",
            ])
            assert run_case(diffusion_input, flags)[0] == pytest.approx(limited_step)
            initial = read_tab(f"tab/{basename}.hydro_3t.00000.tab")
            final = read_tab(f"tab/{basename}.hydro_3t.00001.tab")
            for group in ("erad00", "erad01"):
                reference, coefficients = harmonic_frozen_backward_euler(
                    initial[group], limited_opacity, 1.0, limited_step)
                # The step contains resolved cells whose physical limited coefficient
                # is materially larger than the flat-state regularization.
                assert max(coefficients) > 1.5*(0.5/resolution)
                assert final[group] == pytest.approx(
                    reference, rel=5.0e-8, abs=5.0e-11)
                assert min(final[group]) >= 0.0
                assert sum(final[group]) == pytest.approx(
                    sum(initial[group]), rel=5.0e-10, abs=5.0e-11)

        # A vacuum face uses a zero-radiation ghost, but its frozen harmonic coefficient
        # must still enforce |F|<=alpha*c*E_face.  The cell-centered boundary gradient
        # alone gives a coefficient several times too large for a uniform interior.
        # Compare against the exact nonperiodic backward-Euler matrix, including the
        # face-only alpha*c*dx/2 cap used by the operator and preconditioner.
        vacuum_resolution = 64
        vacuum_step = 5.0e-4
        vacuum_basename = "fld_implicit_harmonic_vacuum"
        vacuum_implicit_input = input_variant(
            diffusion_input, variants, "implicit_harmonic_vacuum",
            {"thermal_radiation": {
                "transport_integrator": "implicit",
                "implicit_x1_inner_boundary": "vacuum",
                "implicit_x1_outer_boundary": "vacuum",
            }})
        vacuum_flags = transport_flags(
            vacuum_basename, resolution=vacuum_resolution,
            opacity=limited_opacity, nlim=1)
        vacuum_flags.extend([
            f"time/tlim={vacuum_step}",
            f"output1/dt={vacuum_step}",
            "mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
            "thermal_radiation/initial_profile=uniform",
            "thermal_radiation/source_cfl=0",
            "thermal_radiation/flux_limiter=harmonic",
            "thermal_radiation/c_light=1.0",
        ])
        assert run_case(vacuum_implicit_input, vacuum_flags)[0] == pytest.approx(
            vacuum_step)
        vacuum_initial = read_tab(
            f"tab/{vacuum_basename}.hydro_3t.00000.tab")
        vacuum_final = read_tab(
            f"tab/{vacuum_basename}.hydro_3t.00001.tab")
        for group in ("erad00", "erad01"):
            reference, coefficients, boundary_cap = (
                harmonic_vacuum_backward_euler(
                    vacuum_initial[group], limited_opacity, 1.0, vacuum_step))
            assert max(coefficients[0], coefficients[-1]) > 1.5*boundary_cap
            assert vacuum_final[group] == pytest.approx(
                reference, rel=5.0e-8, abs=5.0e-11)
            assert min(vacuum_final[group]) >= 0.0
            assert sum(vacuum_final[group]) < sum(vacuum_initial[group])

        # Exercise the production group count.  Every group participates in the face
        # reductions; the causal bound must not acquire an ngroups factor.
        twenty_parameters = {"n_groups": 20}
        for group in range(21):
            twenty_parameters[f"group_bound_{group}"] = 5.0*group
        for group in range(20):
            twenty_parameters[f"kappa_transport_{group}"] = 1.0e-10
        twenty_input = input_variant(
            diffusion_input, variants, "twenty_group",
            {"thermal_radiation": twenty_parameters})
        twenty_group = transport_flags("fld_dt_20group", opacity=1.0e-10)
        assert run_case(twenty_input, twenty_group)[0] == pytest.approx(
            streaming_dt[64], rel=2.0e-6)

        # Native opacity tables must enter the same two asymptotic limits, including
        # values far outside the small example table's nominal scale.
        table_common = [
            f"thermal_radiation/opacity_table_file={opacity_table}",
            "time/nlim=0", "time/tlim=1.0",
            "problem/pl=1.0e-8", "problem/pr=1.0e-8", "output1/dt=1.0",
        ]
        table_thin_input = input_variant(
            table_input, variants, "table_thin",
            {"thermal_radiation": {"opacity_value_scale": 1.0e-12}})
        table_thin = [
            "job/basename=fld_dt_table_thin", *table_common,
        ]
        assert run_case(table_thin_input, table_thin)[0] == pytest.approx(
            streaming_dt[64], rel=2.0e-6)
        table_thick_input = input_variant(
            table_input, variants, "table_thick",
            {"thermal_radiation": {"opacity_value_scale": 1.0e10}})
        table_thick = [
            "job/basename=fld_dt_table_thick", *table_common,
            "thermal_radiation/c_light=1.0e10",
            "thermal_radiation/initial_radiation_temperature_right=0.999999",
        ]
        assert run_case(table_thick_input, table_thick)[0] == pytest.approx(
            expected_diffusion, rel=2.0e-6)

        # The same 1-D solution embedded in 2-D and 3-D must retain its one-dimensional
        # streaming CFL and remain stable for multiple RK2 steps.  AP faces with zero
        # normal flux make no contribution in inactive transverse directions instead of
        # introducing the enormous transverse secant coefficient that collapsed dt.
        multidimensional = ((2, 32, 8), (3, 16, 6))
        for ndim, resolution, nlim in multidimensional:
            basename = f"fld_dt_{ndim}d"
            flags = transport_flags(
                basename, resolution=resolution, nlim=nlim)
            flags.extend([
                f"mesh/nx2={resolution}",
                f"meshblock/nx2={min(16, resolution)}",
            ])
            if ndim == 3:
                flags.extend([
                    f"mesh/nx3={resolution}",
                    f"meshblock/nx3={min(16, resolution)}",
                ])
            timesteps = run_case(diffusion_input, flags)
            expected = 0.4/resolution
            assert timesteps[0] == pytest.approx(expected, rel=2.0e-6)
            assert min(timesteps) > 0.99*expected
            assert_slice_conserved_and_positive(basename)

        # Zero-radiation inflow ghosts form vacuum Dirichlet faces.  Near the radiation
        # floor, the secant flux/energy speed supplies a positivity CFL even though the
        # differential dF/dE term is zero while the floor is active.
        vacuum_input = input_variant(
            diffusion_input, variants, "vacuum_floor",
            {"thermal_radiation": {"energy_floor": 1.0e-12}})
        vacuum = transport_flags("fld_dt_vacuum", nlim=6)
        vacuum.extend([
            "mesh/ix1_bc=inflow", "mesh/ox1_bc=inflow",
            "thermal_radiation/initial_profile=uniform",
            "thermal_radiation/initial_radiation_temperature=1.0e-3",
        ])
        vacuum_timesteps = run_case(vacuum_input, vacuum)
        assert min(vacuum_timesteps) >= 0.99*(0.4/64.0)
        final_vacuum = read_tab("tab/fld_dt_vacuum.hydro_3t.00001.tab")
        assert min(final_vacuum["erad00"]) >= 0.0
        assert min(final_vacuum["erad01"]) >= 0.0

        # The implicit matter-radiation source remains independently constrained by
        # source_cfl, and its local matter+radiation exchange remains conservative.
        source_common = [
            "time/nlim=0", "time/tlim=1.0", "output1/dt=1.0",
            "problem/pl=1.0e-8", "problem/pr=1.0e-8",
        ]
        source_dt = run_case(
            relax_input,
            ["job/basename=fld_dt_source", *source_common])[0]
        assert source_dt == pytest.approx(3.75e-6, rel=2.0e-6)
        implicit_source_input = input_variant(
            relax_input, variants, "implicit_source",
            {"thermal_radiation": {"transport_integrator": "implicit"}})
        implicit_source_dt = run_case(
            implicit_source_input,
            ["job/basename=fld_dt_source_implicit", *source_common])[0]
        assert implicit_source_dt == pytest.approx(source_dt, rel=2.0e-6)
        half_source_dt = run_case(
            relax_input,
            ["job/basename=fld_dt_source_half", *source_common,
             "thermal_radiation/source_cfl=0.05"])[0]
        assert half_source_dt == pytest.approx(0.5*source_dt, rel=2.0e-6)

        source_evolve = [
            "job/basename=fld_dt_source_evolve",
            "time/nlim=1", "time/tlim=1.0", "output1/dt=1.0",
            "problem/pl=1.0e-8", "problem/pr=1.0e-8",
        ]
        run_case(relax_input, source_evolve)
        source_initial = read_tab(
            "tab/fld_dt_source_evolve.hydro_3t.00000.tab")
        source_final = read_tab(
            "tab/fld_dt_source_evolve.hydro_3t.00001.tab")
        for index in range(len(source_initial["erad"])):
            initial_total = (source_initial["eion"][index]
                             + source_initial["eele"][index]
                             + source_initial["erad"][index])
            final_total = (source_final["eion"][index]
                           + source_final["eele"][index]
                           + source_final["erad"][index])
            assert final_total == pytest.approx(
                initial_total, rel=3.0e-11, abs=3.0e-13)
    finally:
        variants_context.cleanup()
        testutils.cleanup()
