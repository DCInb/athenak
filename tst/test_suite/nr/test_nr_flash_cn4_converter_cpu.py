"""Focused CLI tests for the synthetic FLASH CN4-to-AthenaK converter."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
CONVERTER = REPOSITORY / "scripts" / "flash_cn4_to_athenak.py"

AMU_G = 1.66053906660e-24
EV_TO_K = 11604.518121550082
NUMBER_DENSITY_REFERENCE = 1.0e23
PRESSURE_REFERENCE = 1.0e15
GAMMA1 = 5.0 / 3.0
GAMMA3M1 = 2.0 / 3.0
MANUAL_ABAR = 12.0
LOG_ABAR = 20.0
MANUAL_MASS_G = MANUAL_ABAR * AMU_G
LOG_MASS_G = LOG_ABAR * AMU_G

EOS_NAMES = (
    "zbar",
    "dzdt",
    "pion",
    "pele",
    "dpidt",
    "dpedt",
    "eion",
    "eele",
    "cvion",
    "cvele",
    "deidn",
    "deedn",
)


def fortran_e12(value):
    """Encode a finite value as one whitespace-free E12.6-style field."""
    assert math.isfinite(value)
    magnitude = abs(value)
    if magnitude == 0.0:
        exponent = 0
        fraction = 0.0
    else:
        exponent = math.floor(math.log10(magnitude)) + 1
        fraction = magnitude / 10.0**exponent
    mantissa = f"{fraction:.6f}"
    assert mantissa.startswith("0.")
    if value < 0.0:
        mantissa = "-." + mantissa[2:]
    field = f"{mantissa}E{exponent:+03d}"
    assert len(field) == 12
    return field


def integer_field(value):
    """Encode a nonnegative integer in one fixed-width CN4 field."""
    field = f"{value:012d}"
    assert len(field) == 12
    return field


def thermodynamic_values(number_density, temperature_ev, mass_per_ion):
    """Return a monatomic ideal-gas EOS with exact power-law surfaces."""
    pressure = (
        PRESSURE_REFERENCE
        * number_density / NUMBER_DENSITY_REFERENCE
        * temperature_ev
    )
    rho = number_density * mass_per_ion
    energy = pressure / (rho * GAMMA3M1)
    log_density = math.log(number_density / NUMBER_DENSITY_REFERENCE)
    log_temperature = math.log(temperature_ev)
    zbar = 1.0 + 0.05 * log_density + 0.04 * log_temperature
    return pressure, energy, zbar


def eos_blocks(number_density, temperature_ev, mass_per_ion,
               energy_reference_shift=0.0):
    """Construct all twelve temperature-fastest two-temperature EOS blocks."""
    fields = {name: [] for name in EOS_NAMES}
    for density in number_density:
        for temperature in temperature_ev:
            pressure, energy, zbar = thermodynamic_values(
                density, temperature, mass_per_ion
            )
            source_energy = energy - energy_reference_shift
            fields["zbar"].append(zbar)
            fields["dzdt"].append(0.0)
            fields["pion"].append(0.25 * pressure / 1.0e7)
            fields["pele"].append(0.75 * pressure / 1.0e7)
            fields["dpidt"].append(0.0)
            fields["dpedt"].append(0.0)
            fields["eion"].append(0.40 * source_energy / 1.0e7)
            fields["eele"].append(0.60 * source_energy / 1.0e7)
            fields["cvion"].append(0.0)
            fields["cvele"].append(0.0)
            fields["deidn"].append(0.0)
            fields["deedn"].append(0.0)
    return fields


def write_payload(path, ntemperature, ndensity, fields):
    """Write a complete synthetic CN4 file from already encoded fields."""
    lines = [
        f"{ntemperature:10d}{ndensity:10d}",
        "synthetic ion composition",
        "synthetic electron composition",
    ]
    lines.extend(
        "".join(fields[index:index + 4])
        for index in range(0, len(fields), 4)
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_manual_cn4(path, truncate=False, malformed=False,
                     energy_reference_shift=0.0, ngroups=1):
    """Write a descending-axis, manual-grid CN4 table without entropy."""
    temperatures = [100.0, 10.0, 1.0]
    number_density = [1.0e25, 1.0e24, 1.0e23]
    blocks = eos_blocks(
        number_density,
        temperatures,
        MANUAL_MASS_G,
        energy_reference_shift,
    )
    payload = [integer_field(ngroups)]
    payload.extend(fortran_e12(value) for value in temperatures)
    payload.extend(fortran_e12(value) for value in number_density)
    for name in EOS_NAMES:
        payload.extend(fortran_e12(value) for value in blocks[name])
    boundary_count = 2 if ngroups == 0 else ngroups + 1
    payload.extend(fortran_e12(value) for value in range(boundary_count))
    table_size = len(temperatures) * len(number_density)
    for opacity in (1.0, 2.0, 3.0):
        payload.extend(
            fortran_e12(opacity) for _ in range(table_size * ngroups)
        )
    if malformed:
        payload[0] = "not-an-int!!"
    if truncate:
        payload.pop()
    write_payload(path, len(temperatures), len(number_density), payload)


def write_log_cn4(path):
    """Write an increasing logarithmic-grid CN4 table with electron entropy."""
    temperatures = [1.0, 10.0, 100.0]
    number_density = [1.0e22, 1.0e23, 1.0e24]
    blocks = eos_blocks(number_density, temperatures, LOG_MASS_G)
    payload = [
        fortran_e12(1.0),
        fortran_e12(22.0),
        fortran_e12(1.0),
        fortran_e12(0.0),
        integer_field(1),
    ]
    for name in EOS_NAMES:
        payload.extend(fortran_e12(value) for value in blocks[name])
    table_size = len(temperatures) * len(number_density)
    payload.extend(fortran_e12(5.0 + index) for index in range(table_size))
    payload.extend(fortran_e12(value) for value in (0.0, 1.0))
    for opacity in (1.0, 2.0, 3.0):
        payload.extend(fortran_e12(opacity) for _ in range(table_size))
    write_payload(path, len(temperatures), len(number_density), payload)


def replace_payload_fields(path, replacements):
    """Replace selected fixed-width payload tokens without changing layout."""
    lines = path.read_text(encoding="ascii").splitlines()
    payload_text = "".join(lines[3:])
    assert len(payload_text) % 12 == 0
    fields = [
        payload_text[index:index + 12]
        for index in range(0, len(payload_text), 12)
    ]
    for index, value in replacements.items():
        assert len(value) == 12
        fields[index] = value
    rewritten = lines[:3]
    rewritten.extend(
        "".join(fields[index:index + 4])
        for index in range(0, len(fields), 4)
    )
    path.write_text("\n".join(rewritten) + "\n", encoding="ascii")


def run_converter(input_path, output_path, *options):
    """Invoke the converter through the public command-line interface."""
    command = [
        sys.executable,
        str(CONVERTER),
        str(input_path),
    ]
    if output_path is not None:
        command.append(str(output_path))
    command.extend(options)
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def load_converter_module():
    """Load the converter for parser and in-memory edge-case checks."""
    module_name = "flash_cn4_to_athenak_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    specification = importlib.util.spec_from_file_location(
        module_name, CONVERTER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def in_memory_ionmix_table(module, number_density):
    """Build a valid parsed table around a caller-selected density axis."""
    ndensity = len(number_density)
    temperatures = [1.0, 10.0, 100.0]
    zero = [[0.0] * len(temperatures) for _ in range(ndensity)]
    increasing = [
        [float(index + 1) for index in range(len(temperatures))]
        for _ in range(ndensity)
    ]
    fields = {
        name: [list(row) for row in zero]
        for name in EOS_NAMES
    }
    fields["zbar"] = [[1.0] * len(temperatures) for _ in range(ndensity)]
    fields["pion"] = [list(row) for row in increasing]
    fields["eion"] = [list(row) for row in increasing]
    return module.IonmixTable(
        ntemperature=len(temperatures),
        ndensity=ndensity,
        ngroups=0,
        grid_mode="manual",
        has_electron_entropy=False,
        temperature_ev=temperatures,
        number_density_cm3=list(number_density),
        fields=fields,
    )


def parse_native_table(path):
    """Parse the small native-v2 subset emitted by the converter."""
    tokens = []
    for line in path.read_text(encoding="ascii").splitlines():
        tokens.extend(line.partition("#")[0].split())
    cursor = 0

    def take(label=None):
        nonlocal cursor
        token = tokens[cursor]
        cursor += 1
        if label is not None:
            assert token == label
        return token

    take("athenak_eos_table")
    assert take() == "2"
    take("dimensions")
    ndensity = int(take())
    ntemperature = int(take())
    count = ndensity * ntemperature
    take("density")
    density = [float(take()) for _ in range(ndensity)]
    take("temperature")
    temperature = [float(take()) for _ in range(ntemperature)]
    result = {"density": density, "temperature": temperature}
    for name in (
        "pressure",
        "specific_internal_energy",
        "sound_speed_squared",
    ):
        take(name)
        result[name] = [float(take()) for _ in range(count)]
    take("material_fields")
    material_count = int(take())
    for _ in range(material_count):
        name = take()
        result[name] = [float(take()) for _ in range(count)]
    take("end")
    assert cursor == len(tokens)
    result["dimensions"] = (ndensity, ntemperature)
    return result


def expected_surfaces(number_density, temperature_ev, mass_per_ion):
    """Return independently derived AthenaK fields on increasing axes."""
    pressure = []
    energy = []
    sound_speed_squared = []
    gamma1 = []
    gamma3m1 = []
    zbar = []
    for density_number in number_density:
        rho = density_number * mass_per_ion
        for temperature in temperature_ev:
            p_value, e_value, zbar_value = thermodynamic_values(
                density_number, temperature, mass_per_ion
            )
            cs2_value = GAMMA1 * p_value / rho
            pressure.append(p_value)
            energy.append(e_value)
            sound_speed_squared.append(cs2_value)
            gamma1.append(GAMMA1)
            gamma3m1.append(GAMMA3M1)
            zbar.append(zbar_value)
    return {
        "pressure": pressure,
        "specific_internal_energy": energy,
        "sound_speed_squared": sound_speed_squared,
        "gamma1": gamma1,
        "gamma3m1": gamma3m1,
        "zbar": zbar,
    }


def assert_converted_fields(table, number_density, temperature_ev,
                            mass_per_ion, abar):
    """Check axes, CGS conversion, thermodynamic derivatives, and materials."""
    assert table["dimensions"] == (len(number_density), len(temperature_ev))
    expected_density = [value * mass_per_ion for value in number_density]
    expected_temperature = [value * EV_TO_K for value in temperature_ev]
    assert table["density"] == pytest.approx(expected_density, rel=2.0e-12)
    assert table["temperature"] == pytest.approx(
        expected_temperature, rel=2.0e-12
    )

    expected = expected_surfaces(
        number_density, temperature_ev, mass_per_ion
    )
    assert table["pressure"] == pytest.approx(
        expected["pressure"], rel=3.0e-6
    )
    assert table["specific_internal_energy"] == pytest.approx(
        expected["specific_internal_energy"], rel=5.0e-6
    )
    assert table["zbar"] == pytest.approx(expected["zbar"], rel=5.0e-6)
    for name in ("sound_speed_squared", "gamma1", "gamma3m1"):
        assert table[name] == pytest.approx(expected[name], rel=3.0e-5)
    assert table["abar"] == pytest.approx(
        [abar] * (len(number_density) * len(temperature_ev)), rel=2.0e-12
    )


def test_manual_grid_cli_converts_units_and_reorders_axes(tmp_path):
    input_path = tmp_path / "manual.cn4"
    output_path = tmp_path / "manual.eos"
    write_manual_cn4(input_path)

    result = run_converter(
        input_path,
        output_path,
        "--abar",
        "12",
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--quiet",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    table = parse_native_table(output_path)
    assert_converted_fields(
        table,
        [1.0e23, 1.0e24, 1.0e25],
        [1.0, 10.0, 100.0],
        MANUAL_MASS_G,
        MANUAL_ABAR,
    )


def test_log_grid_cli_detects_electron_entropy(tmp_path):
    input_path = tmp_path / "log.cn4"
    output_path = tmp_path / "log.eos"
    write_log_cn4(input_path)
    mass_per_ion = LOG_MASS_G

    result = run_converter(
        input_path,
        output_path,
        "--mass-per-ion",
        f"{mass_per_ion:.17e}",
        "--grid-mode",
        "log",
    )

    assert result.returncode == 0, result.stderr
    assert "log grid, with electron entropy" in result.stdout
    table = parse_native_table(output_path)
    assert_converted_fields(
        table,
        [1.0e22, 1.0e23, 1.0e24],
        [1.0, 10.0, 100.0],
        mass_per_ion,
        LOG_ABAR,
    )


def test_omitted_e_three_digit_fortran_exponents(tmp_path):
    input_path = tmp_path / "omitted_e.cn4"
    output_path = tmp_path / "omitted_e.eos"
    write_manual_cn4(input_path)
    replace_payload_fields(
        input_path,
        {
            16: "0.100000+101",
            17: "0.100000-099",
            18: "-.100000-099",
        },
    )

    module = load_converter_module()
    _, _, tokens = module._read_cn4_tokens(input_path)
    parsed_values = [
        module._token_float(tokens, index, "test omitted-E value")
        for index in (16, 17, 18)
    ]
    assert parsed_values == pytest.approx(
        [1.0e100, 1.0e-100, -1.0e-100], rel=1.0e-14, abs=0.0
    )
    parsed = module.read_ionmix4(
        input_path, grid_mode="manual", electron_entropy="absent"
    )
    assert parsed.grid_mode == "manual"

    result = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--quiet",
    )
    assert result.returncode == 0, result.stderr


def test_zero_opacity_groups_accepts_two_opacplot2_dummy_bounds(tmp_path):
    input_path = tmp_path / "zero_groups.cn4"
    output_path = tmp_path / "zero_groups.eos"
    write_manual_cn4(input_path, ngroups=0)

    result = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--quiet",
    )

    assert result.returncode == 0, result.stderr
    assert parse_native_table(output_path)["dimensions"] == (3, 3)


def test_explicit_energy_offset_restores_reference_and_derivatives(tmp_path):
    input_path = tmp_path / "shifted_energy.cn4"
    output_path = tmp_path / "shifted_energy.eos"
    energy_shift = 1.0e16
    write_manual_cn4(input_path, energy_reference_shift=energy_shift)

    failed = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
    )
    assert failed.returncode == 2
    assert "--energy-offset-erg-g" in failed.stderr
    assert not output_path.exists()

    converted = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--energy-offset-erg-g",
        f"{energy_shift:.17e}",
        "--quiet",
    )
    assert converted.returncode == 0, converted.stderr
    assert "Specific-energy reference offset" in output_path.read_text(
        encoding="ascii"
    )
    table = parse_native_table(output_path)
    assert_converted_fields(
        table,
        [1.0e23, 1.0e24, 1.0e25],
        [1.0, 10.0, 100.0],
        MANUAL_MASS_G,
        MANUAL_ABAR,
    )


def test_validate_only_writes_no_output(tmp_path):
    input_path = tmp_path / "validate.cn4"
    write_manual_cn4(input_path)

    result = run_converter(
        input_path,
        None,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--validate-only",
    )

    assert result.returncode == 0, result.stderr
    assert f"validated {input_path}" in result.stdout
    assert list(tmp_path.iterdir()) == [input_path]


def test_existing_output_is_not_clobbered_without_force(tmp_path):
    input_path = tmp_path / "no_clobber.cn4"
    output_path = tmp_path / "no_clobber.eos"
    write_manual_cn4(input_path)
    output_path.write_bytes(b"keep this exact content\n")

    result = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--quiet",
    )

    assert result.returncode == 2
    assert "output already exists" in result.stderr
    assert output_path.read_bytes() == b"keep this exact content\n"


def test_oversized_dimensions_fail_before_payload_read(tmp_path):
    input_path = tmp_path / "oversized.cn4"
    input_path.write_text(
        f"{20000:10d}{20000:10d}\nion header\nelectron header\n",
        encoding="ascii",
    )

    result = run_converter(
        input_path,
        None,
        "--abar",
        str(MANUAL_ABAR),
        "--validate-only",
    )

    assert result.returncode == 2
    assert "no numeric payload" not in result.stderr
    assert "9*NDENS*NTEMP" in result.stderr or "C++ int" in result.stderr


def test_collapsed_converted_and_log_density_axes_fail_cleanly():
    module = load_converter_module()
    adjacent = [1.0]
    adjacent.append(math.nextafter(adjacent[-1], math.inf))
    adjacent.append(math.nextafter(adjacent[-1], math.inf))
    converted_collapse = in_memory_ionmix_table(module, adjacent)
    with pytest.raises(
        module.ConversionError,
        match=r"mass density.*strictly increasing",
    ):
        module.convert_to_athenak(converted_collapse, math.ulp(0.0))

    huge = [1.0e300]
    huge.append(math.nextafter(huge[-1], math.inf))
    huge.append(math.nextafter(huge[-1], math.inf))
    log_collapse = in_memory_ionmix_table(module, huge)
    with pytest.raises(
        module.ConversionError,
        match=r"log.*density.*strictly increasing",
    ):
        module.convert_to_athenak(log_collapse, 1.0)


@pytest.mark.parametrize(
    ("fixture_options", "diagnostic"),
    (
        ({"truncate": True}, "payload has"),
        ({"malformed": True}, "opacity group count"),
    ),
)
def test_invalid_cn4_fails_without_output(tmp_path, fixture_options,
                                          diagnostic):
    input_path = tmp_path / "invalid.cn4"
    output_path = tmp_path / "invalid.eos"
    write_manual_cn4(input_path, **fixture_options)

    result = run_converter(
        input_path,
        output_path,
        "--abar",
        "12",
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
    )

    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert not output_path.exists()


def test_generated_table_loads_in_athenak_when_binary_is_available(tmp_path):
    athena = Path.cwd() / "athena"
    if not athena.is_file():
        pytest.skip("./athena is available only in the custom build runner")

    input_path = tmp_path / "loader.cn4"
    output_path = tmp_path / "loader.eos"
    write_manual_cn4(input_path)
    converted = run_converter(
        input_path,
        output_path,
        "--abar",
        str(MANUAL_ABAR),
        "--grid-mode",
        "manual",
        "--electron-entropy",
        "absent",
        "--quiet",
    )
    assert converted.returncode == 0, converted.stderr

    source_deck = REPOSITORY / "inputs" / "hydro" / "tabulated_eos.athinput"
    loader_deck = tmp_path / "loader.athinput"
    loader_deck.write_text(
        source_deck.read_text(encoding="ascii")
        + "\n<units>\n"
        + "length_cgs = 1.0\n"
        + "mass_cgs = 1.0\n"
        + "time_cgs = 1.0\n"
        + "mu = 1.0\n",
        encoding="ascii",
    )
    command = [
        str(athena),
        "-i",
        str(loader_deck),
        "-d",
        str(tmp_path / "run"),
        "job/basename=cn4_loader_smoke",
        f"hydro/table_file={output_path}",
        "hydro/table_unit_system=cgs",
        "hydro/table_bounds=clamp",
        "time/nlim=0",
        "time/tlim=0.0",
        "output1/dt=-1.0",
        "output2/dt=-1.0",
    ]
    loaded = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
