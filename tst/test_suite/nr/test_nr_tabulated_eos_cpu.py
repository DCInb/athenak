"""Regression test for the portable Newtonian density-temperature EOS table."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/hydro/tabulated_eos.athinput"
mhd_input_file = "../../../inputs/mhd/tabulated_eos.athinput"
table_file = "../../../inputs/hydro/gamma_law_eos_table.dat"
binary_table_file = "gamma_law_eos_tablereader.dat"
non_gamma_table_file = "power_law_eos_table.dat"
negative_size_table_file = "negative_size_eos_tablereader.dat"
overflow_size_table_file = "overflow_size_eos_tablereader.dat"
truncated_table_file = "truncated_eos_tablereader.dat"


def tablereader_header(density_size, temperature_size):
    """Return a minimal TableReader header for the portable EOS fields."""
    return (
        "<metadatabegin>\n"
        "endianness = little\n"
        "log_axis_base = e\n"
        "<metadataend>\n"
        "<scalarsbegin>\n"
        "<scalarsend>\n"
        "<pointsbegin>\n"
        f"logrho = {density_size}\n"
        f"logtemp = {temperature_size}\n"
        "<pointsend>\n"
        "<fieldsbegin>\n"
        "logpress\n"
        "logeps\n"
        "logcs2\n"
        "<fieldsend>\n"
    )


def write_tablereader_table(filename):
    """Write the same gamma-law EOS using AthenaK's binary TableReader format."""
    gamma = 1.4
    density = np.array([0.01, 0.1, 1.0, 10.0])
    temperature = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    rho_grid = density[:, None]
    temperature_grid = np.broadcast_to(
        temperature[None, :], (density.size, temperature.size))
    fields = (
        np.log(rho_grid*temperature_grid),
        np.log(temperature_grid/(gamma-1.0)),
        np.log(gamma*temperature_grid),
    )
    header = tablereader_header(density.size, temperature.size)
    with open(filename, "wb") as table:
        table.write(header.encode("ascii"))
        table.write(np.asarray(np.log(density), dtype="<f8").tobytes())
        table.write(np.asarray(np.log(temperature), dtype="<f8").tobytes())
        for field in fields:
            table.write(np.asarray(field, dtype="<f8").tobytes(order="C"))


def write_power_law_table(filename):
    """Write a density-dependent EOS whose log interpolation is analytic."""
    density = np.array([0.1, 1.0, 10.0])
    temperature = np.array([0.1, 1.0, 10.0])
    rho_grid = density[:, None]
    temperature_grid = temperature[None, :]
    fields = (
        rho_grid**1.2 * temperature_grid**0.7,
        rho_grid**0.4 * temperature_grid**1.3,
        1.1 * rho_grid**0.2 * temperature_grid**0.8,
    )
    lines = [
        "athenak_eos_table 1",
        f"dimensions {density.size} {temperature.size}",
        "density " + " ".join(f"{value:.17e}" for value in density),
        "temperature " + " ".join(
            f"{value:.17e}" for value in temperature),
    ]
    labels = (
        "pressure", "specific_internal_energy", "sound_speed_squared")
    for label, field in zip(labels, fields):
        lines.append(label)
        lines.extend(
            " ".join(f"{value:.17e}" for value in row) for row in field)
    lines.append("end")
    with open(filename, "w", encoding="ascii") as table:
        table.write("\n".join(lines) + "\n")


def invalid_table_command(filename):
    """Build the command used to assert an initialization failure."""
    return [
        "./athena", "-i", input_file,
        "job/basename=invalid_eos_table",
        f"hydro/table_file={filename}",
        "time/nlim=0",
        "output1/dt=-1.0",
        "output2/dt=-1.0",
    ]


def test_run():
    try:
        write_tablereader_table(binary_table_file)
        common = [
            "time/tlim=0.05",
            "output1/dt=0.05",
            "output2/dt=-1.0",
        ]
        assert testutils.run(input_file, flags=[
            "job/basename=eos_table",
            f"hydro/table_file={table_file}",
        ] + common), "Tabulated-EOS shock tube failed."
        table = athena_read.tab("tab/eos_table.hydro_w.00001.tab")

        assert testutils.run(input_file, flags=[
            "job/basename=eos_ideal_reference",
            "hydro/eos=ideal",
        ] + common), "Ideal-gas reference shock tube failed."
        reference = athena_read.tab(
            "tab/eos_ideal_reference.hydro_w.00001.tab")

        assert table.keys() == reference.keys()
        for field in table:
            assert np.all(np.isfinite(table[field]))
            assert np.allclose(table[field], reference[field],
                               rtol=2.0e-10, atol=2.0e-12), field

        assert testutils.run(input_file, flags=[
            "job/basename=eos_tablereader",
            f"hydro/table_file={binary_table_file}",
        ] + common), "TableReader-format EOS shock tube failed."
        binary_table = athena_read.tab(
            "tab/eos_tablereader.hydro_w.00001.tab")
        assert binary_table.keys() == table.keys()
        for field in binary_table:
            assert np.all(np.isfinite(binary_table[field]))
            assert np.allclose(binary_table[field], table[field],
                               rtol=2.0e-10, atol=2.0e-12), field

        # Exercise interpolation and inversion with genuine density dependence.
        write_power_law_table(non_gamma_table_file)
        density = np.sqrt(0.1)
        temperature = np.sqrt(0.1)
        pressure = density**1.2 * temperature**0.7
        eint_density = density * density**0.4 * temperature**1.3
        assert testutils.run(input_file, flags=[
            "job/basename=eos_power_law",
            f"hydro/table_file={non_gamma_table_file}",
            "hydro/table_bounds=error",
            f"problem/dl={density:.17e}",
            f"problem/dr={density:.17e}",
            f"problem/pl={pressure:.17e}",
            f"problem/pr={pressure:.17e}",
            "time/tlim=1.0e-4",
            "output1/dt=1.0e-4",
            "output2/dt=-1.0",
        ]), "Density-dependent table EOS run failed."
        power_law = athena_read.tab(
            "tab/eos_power_law.hydro_w.00001.tab")
        assert np.allclose(power_law["dens"], density,
                           rtol=2.0e-10, atol=2.0e-12)
        assert np.allclose(power_law["eint"], eint_density,
                           rtol=2.0e-10, atol=2.0e-12)

        # Hostile dimensions and short payloads must fail before allocation/use.
        with open(negative_size_table_file, "wb") as table_file_handle:
            table_file_handle.write(tablereader_header(-2, 3).encode("ascii"))
        assert not testutils.run_command(
            invalid_table_command(negative_size_table_file))

        with open(overflow_size_table_file, "wb") as table_file_handle:
            maximum_count = np.iinfo(np.int64).max
            table_file_handle.write(
                tablereader_header(maximum_count, maximum_count).encode("ascii"))
        assert not testutils.run_command(
            invalid_table_command(overflow_size_table_file))

        with open(truncated_table_file, "wb") as table_file_handle:
            table_file_handle.write(tablereader_header(2, 3).encode("ascii"))
            table_file_handle.write(np.asarray([0.0], dtype="<f8").tobytes())
        assert not testutils.run_command(
            invalid_table_command(truncated_table_file))

        assert testutils.run(mhd_input_file, flags=[
            "job/basename=eos_table_mhd",
            f"mhd/table_file={table_file}",
        ] + common), "Tabulated-EOS MHD shock tube failed."
        table_mhd = athena_read.tab("tab/eos_table_mhd.mhd_w.00001.tab")

        assert testutils.run(mhd_input_file, flags=[
            "job/basename=eos_ideal_mhd_reference",
            "mhd/eos=ideal",
        ] + common), "Ideal-gas MHD reference shock tube failed."
        reference_mhd = athena_read.tab(
            "tab/eos_ideal_mhd_reference.mhd_w.00001.tab")

        assert table_mhd.keys() == reference_mhd.keys()
        for field in table_mhd:
            assert np.all(np.isfinite(table_mhd[field]))
            assert np.allclose(table_mhd[field], reference_mhd[field],
                               rtol=2.0e-10, atol=2.0e-12), field
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
