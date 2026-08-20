"""Shared fixtures for mixed-material thermal-radiation regressions."""

from pathlib import Path
import re
import sys

import numpy as np

import test_suite.testutils as testutils


NGROUPS = 20
GROUP_BOUNDS = [0.5*group for group in range(NGROUPS + 1)]
SOURCE_DT = 1.0e-4
BASE_INPUT = Path("../../../inputs/mhd/two_material_relax.athinput")


def opacity_value(kind, material, group, density,
                  transport_profile="alternating"):
    """Analytic table values; density dependence exposes partial-density lookup."""
    scale = group + 1.0
    if kind == "transport":
        # Alternate between streaming and diffusion-dominated groups.  Distinct
        # material and partial-density factors make a prepared mixed-table face
        # location observable in transport instead of giving CH and He identical
        # opacities in every group.
        if transport_profile == "last-group-ap":
            regime = 1.0e-8 if group == NGROUPS-1 else 50.0
        else:
            regime = 1.0e-8 if group % 2 == 0 else 20.0
        if material == 0:
            return regime*scale*(1.0 + density)
        return 1.5*regime*scale*(1.0 + 2.0*density)
    if kind == "absorption":
        if material == 0:
            return 0.02*scale*(1.0 + density)
        return 0.03*scale*(1.0 + 2.0*density)
    if material == 0:
        return 0.025*scale*(1.0 + 0.5*density)
    return 0.015*scale*(1.0 + 1.5*density)


def mixed_opacity(kind, group, density, material0_fraction):
    """FLASH partial-density, mass-weighted additive opacity."""
    y0 = min(max(material0_fraction, 0.0), 1.0)
    result = 0.0
    if y0 > 0.0:
        result += y0*opacity_value(kind, 0, group, density*y0)
    if y0 < 1.0:
        result += (1.0-y0)*opacity_value(
            kind, 1, group, density*(1.0-y0))
    return result


def write_opacity_table(path, material, transport_profile="alternating"):
    """Write a 20-group, two-density table with temperature-independent values."""
    densities = (0.1, 1.0)
    temperatures = (0.1, 10.0)
    lines = [
        "athenak_opacity_table 1",
        f"dimensions 2 2 {NGROUPS}",
        "density " + " ".join(str(value) for value in densities),
        "temperature " + " ".join(str(value) for value in temperatures),
        "group_bound " + " ".join(str(value) for value in GROUP_BOUNDS),
    ]
    for kind in ("transport", "absorption", "emission"):
        lines.append(kind)
        for group in range(NGROUPS):
            values = []
            for density in densities:
                value = opacity_value(
                    kind, material, group, density, transport_profile)
                values.extend((value, value))
            lines.append(" ".join(f"{value:.17e}" for value in values))
    lines.extend(("end", ""))
    Path(path).write_text("\n".join(lines), encoding="ascii")


def _update_block(text, block, parameters):
    match = re.search(rf"(?ms)^<{re.escape(block)}>\s*$.*?(?=^<|\Z)", text)
    if match is None:
        raise AssertionError(f"Input block <{block}> not found")
    block_text = match.group(0)
    for key, value in parameters.items():
        pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
        replacement = f"{key} = {value}"
        if re.search(pattern, block_text):
            block_text = re.sub(pattern, replacement, block_text, count=1)
        else:
            block_text = block_text.rstrip() + "\n" + replacement + "\n\n"
    return text[:match.start()] + block_text + text[match.end():]


def write_mixed_input(path, material0_table, material1_table):
    """Build a uniform CH/He source-coupling case around the stock material deck."""
    text = BASE_INPUT.read_text(encoding="ascii")
    text = _update_block(text, "time", {
        "integrator": "rk1", "nlim": 1, "tlim": SOURCE_DT,
        "initial_dt": SOURCE_DT,
    })
    text = _update_block(text, "mhd", {
        "initial_electron_temperature_ratio": 0.25,
        "t_ei_model": "constant", "t_ei": -1.0,
    })
    text = _update_block(text, "materials", {
        "material0_t_ei": -1.0,
        "material1_name": "He", "material1_abar": 4.0,
        "material1_zbar": 2.0, "material1_zeff": 2.0,
        "material1_t_ei": -1.0,
        "material0_opacity_table_file": material0_table,
        "material1_opacity_table_file": material1_table,
        "material0_opacity_interpolation": "linear",
        "material1_opacity_interpolation": "linear",
        "material0_opacity_coordinate_interpolation": "linear",
        "material1_opacity_coordinate_interpolation": "linear",
    })
    text = _update_block(text, "output1", {
        "variable": "mhd_3t", "data_format": "%20.16e", "dt": SOURCE_DT,
    })

    radiation = [
        "", "<thermal_radiation>", "enabled = true",
        f"n_groups = {NGROUPS}", "arad = 0.1", "c_light = 1.0",
        "initial_profile = uniform", "initial_radiation_temperature = 0.1",
        "initial_radiation_temperature_right = 0.05",
        "initial_radiation_x1 = 0.0",
        "flux_limiter = levermore-pomraning", "flux_limit_coefficient = 1.0",
        "transport_discretization = asymptotic-preserving",
        "ap_streaming_threshold = 0.5", "ap_optical_depth_threshold = 1.0",
        "source_cfl = 0.0", "source_integrator = lagged", "source_report = false",
        "couple_matter = true", "opacity_model = table",
    ]
    radiation.extend(
        f"group_bound_{group} = {bound}"
        for group, bound in enumerate(GROUP_BOUNDS)
    )
    radiation.append("")
    binary_output = [
        "", "<output3>", "file_type = bin", "variable = mhd_3t",
        "dt = -1.0", "",
    ]
    Path(path).write_text(
        text.rstrip() + "\n" + "\n".join(radiation + binary_output),
        encoding="ascii")


def prepare_case(input_path, material0_table, material1_table):
    write_opacity_table(material0_table, 0)
    write_opacity_table(material1_table, 1)
    write_mixed_input(input_path, str(material0_table), str(material1_table))


def _assemble_uniform_field(file_data, field):
    """Assemble a uniform-grid binary field in (x3,x2,x1) index order."""
    result = np.empty(
        (file_data["Nx3"], file_data["Nx2"], file_data["Nx1"]),
        dtype=np.asarray(file_data["mb_data"][field][0]).dtype)
    for location, block in zip(file_data["mb_logical"],
                               file_data["mb_data"][field]):
        values = np.asarray(block)
        nk, nj, ni = values.shape
        i0 = int(location[0])*file_data["nx1_mb"]
        j0 = int(location[1])*file_data["nx2_mb"]
        k0 = int(location[2])*file_data["nx3_mb"]
        result[k0:k0+nk, j0:j0+nj, i0:i0+ni] = values
    return result


def run_mixed_transport_probe(input_path, basename):
    """Exercise all three batched 20-group mixed-opacity AP flux kernels."""
    # Import the repository reader by absolute path: pytest executes from the
    # out-of-source build directory, not beside this fixture module.
    reader_path = Path(__file__).resolve().parents[3]/"vis"/"python"
    sys.path.insert(0, str(reader_path))
    import bin_convert  # pylint: disable=import-outside-toplevel

    output_prefix = Path("bin")/f"{basename}.mhd_3t"
    initial_path = Path(f"{output_prefix}.00000.bin")
    final_path = Path(f"{output_prefix}.00001.bin")
    initial_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)

    flags = [
        f"job/basename={basename}",
        "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8",
        "meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4",
        "mesh/ix2_bc=inflow", "mesh/ox2_bc=inflow",
        "mesh/ix3_bc=inflow", "mesh/ox3_bc=inflow",
        "time/integrator=rk1", "time/nlim=1", f"time/tlim={SOURCE_DT}",
        f"time/initial_dt={SOURCE_DT}",
        "problem/yl=1.0", "problem/yr=0.0",
        "output1/dt=-1.0", "output2/dt=-1.0", f"output3/dt={SOURCE_DT}",
        "thermal_radiation/couple_matter=false",
        "thermal_radiation/source_cfl=0.0",
        "thermal_radiation/c_light=100.0",
        "thermal_radiation/transport_discretization=asymptotic-preserving",
        "thermal_radiation/initial_profile=step",
        "thermal_radiation/initial_radiation_temperature=1.0",
        "thermal_radiation/initial_radiation_temperature_right=0.5",
        "thermal_radiation/initial_radiation_x1=0.0",
    ]
    assert testutils.run(str(input_path), flags=flags), (
        "Three-dimensional mixed-opacity transport probe failed.")
    assert initial_path.is_file() and final_path.is_file()

    initial_raw = bin_convert.read_binary(str(initial_path))
    final_raw = bin_convert.read_binary(str(final_path))
    assert (initial_raw["Nx1"], initial_raw["Nx2"], initial_raw["Nx3"]) == (8, 8, 8)
    assert initial_raw["cycle"] == 0
    assert final_raw["cycle"] == 1
    assert final_raw["time"] == SOURCE_DT

    group_fields = [f"erad{group:02d}" for group in range(NGROUPS)]
    assert all(field in initial_raw["var_names"] for field in group_fields)
    initial = {}
    final = {}
    for field in group_fields:
        initial[field] = _assemble_uniform_field(initial_raw, field)
        final[field] = _assemble_uniform_field(final_raw, field)
        assert np.all(np.isfinite(final[field])), field
        assert np.all(final[field] >= 0.0), field
        # Trad=1/0.5 and bounds ending at 10 keep every group populated;
        # each one must therefore traverse the new inner group loop.
        assert np.all(initial[field] > 0.0), field
        assert np.any(final[field] != initial[field]), field

    # The x1 temperature step supplies an interior flux, while zero-radiation
    # inflow ghosts on x2/x3 supply deterministic transverse fluxes.  Probe a
    # streaming and a diffusion-dominated group away from edges shared by two
    # boundary conditions.
    for field in ("erad00", "erad01"):
        delta = final[field]-initial[field]
        assert np.any(delta[4, 4, :] != 0.0), f"missing x1 flux for {field}"
        assert delta[4, 0, 2] != 0.0, f"missing x2 flux for {field}"
        assert delta[0, 4, 2] != 0.0, f"missing x3 flux for {field}"

    # Odd groups are optically thick and retain the composition dependence of
    # the prepared face opacity.  Symmetric x2 boundary cells in pure CH and
    # pure He must consequently have different fractional transport updates.
    thick = "erad01"
    ch_loss = (initial[thick][4, 0, 2]-final[thick][4, 0, 2]) / initial[thick][4, 0, 2]
    he_loss = (initial[thick][4, 0, 5]-final[thick][4, 0, 5]) / initial[thick][4, 0, 5]
    assert ch_loss > 0.0 and he_loss > 0.0
    assert not np.isclose(ch_loss, he_loss, rtol=1.0e-3, atol=0.0)

    return initial_path, final_path


def electron_heat_capacity_fraction(material0_fraction):
    y0 = material0_fraction
    ion_weight = y0/6.5 + (1.0-y0)/4.0
    electron_weight = y0*3.5/6.5 + (1.0-y0)*2.0/4.0
    return electron_weight/(ion_weight+electron_weight)
