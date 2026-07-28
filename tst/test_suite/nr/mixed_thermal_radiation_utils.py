"""Shared fixtures for mixed-material thermal-radiation regressions."""

from pathlib import Path
import re


NGROUPS = 20
GROUP_BOUNDS = [0.5*group for group in range(NGROUPS + 1)]
SOURCE_DT = 1.0e-4
BASE_INPUT = Path("../../../inputs/mhd/two_material_relax.athinput")


def opacity_value(kind, material, group, density):
    """Analytic table values; density dependence exposes partial-density lookup."""
    scale = group + 1.0
    if kind == "transport":
        return 1.0e-8*scale
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


def write_opacity_table(path, material):
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
                value = opacity_value(kind, material, group, density)
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
        "source_cfl = 0.0", "couple_matter = true", "opacity_model = table",
    ]
    radiation.extend(
        f"group_bound_{group} = {bound}"
        for group, bound in enumerate(GROUP_BOUNDS)
    )
    radiation.append("")
    Path(path).write_text(text.rstrip() + "\n" + "\n".join(radiation),
                          encoding="ascii")


def prepare_case(input_path, material0_table, material1_table):
    write_opacity_table(material0_table, 0)
    write_opacity_table(material1_table, 1)
    write_mixed_input(input_path, str(material0_table), str(material1_table))


def electron_heat_capacity_fraction(material0_fraction):
    y0 = material0_fraction
    ion_weight = y0/6.5 + (1.0-y0)/4.0
    electron_weight = y0*3.5/6.5 + (1.0-y0)*2.0/4.0
    return electron_weight/(ion_weight+electron_weight)
