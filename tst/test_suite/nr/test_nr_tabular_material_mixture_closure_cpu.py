"""CPU device test for unequal-grid tabular material mixing and LLF pressure."""

from pathlib import Path
import shutil

import pytest

import test_suite.testutils as testutils


test_directory = Path(__file__).resolve().parent
source_root = test_directory.parents[2]
driver = test_directory / "tabular_material_mixture_driver.cpp"
ch_table = test_directory / "tabular_material_mixture_ch_fixture.dat"
he_table = test_directory / "tabular_material_mixture_he_fixture.dat"
unit_build = Path("tabular_material_mixture_closure_unit_build")


def test_cached_flux_and_exchange_sources():
    flux_source = (source_root / "src/mhd/mhd_fluxes.cpp").read_text(
        encoding="utf-8")
    llf_source = (source_root / "src/mhd/rsolvers/llf_mhd.hpp").read_text(
        encoding="utf-8")
    exchange_source = (
        source_root / "src/two_temperature/two_temperature.cpp").read_text(
            encoding="utf-8")
    radiation_source = (
        source_root / "src/two_temperature/thermal_radiation.cpp").read_text(
            encoding="utf-8")
    assert "ptwo_temp->thermodynamics" in flux_source
    assert "ReconstructMaterialThermodynamicsX1" in flux_source
    assert "ReconstructMaterialThermodynamicsX2" in flux_source
    assert "ReconstructMaterialThermodynamicsX3" in flux_source
    assert "StateFromRhoSpecificEnergies" not in llf_source
    assert "StateFromRhoTotalEnergyTemperatureDifference" in exchange_source
    assert "iteration < 32" not in exchange_source
    coupling_tail = exchange_source.split("pradiation->Couple", 1)[1]
    assert "RefreshMaterialThermodynamics" in coupling_tail
    assert "mixture.MinimumState(density, y0)" in radiation_source
    assert "eele_old-eele_floor-negative" in radiation_source


def build_driver():
    athena_build = Path.cwd().parent
    kokkos_package = athena_build / "cmake_packages" / "Kokkos"
    cmake_source = unit_build / "source"
    cmake_source.mkdir(parents=True)
    cmake_text = f"""
cmake_minimum_required(VERSION 3.16)
project(tabular_material_mixture_closure_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Kokkos CONFIG REQUIRED
             PATHS \"{kokkos_package}\" NO_DEFAULT_PATH)
add_executable(tabular_material_mixture_driver
  \"{driver}\"
  \"{source_root / 'src/materials/ionmix_two_temperature_table.cpp'}\")
target_include_directories(tabular_material_mixture_driver PRIVATE
  \"{source_root / 'src'}\" \"{athena_build}\")
target_compile_definitions(tabular_material_mixture_driver PRIVATE
  KOKKOS_DEPENDENCE)
target_link_libraries(tabular_material_mixture_driver PRIVATE Kokkos::kokkos)
"""
    (cmake_source / "CMakeLists.txt").write_text(cmake_text, encoding="ascii")
    assert testutils.run_command([
        "cmake", "-S", str(cmake_source), "-B", str(unit_build / "build")])
    assert testutils.run_command([
        "cmake", "--build", str(unit_build / "build"), "--parallel", "2"])
    return unit_build / "build" / "tabular_material_mixture_driver"


def test_run():
    try:
        executable = build_driver()
        assert testutils.run_command([
            str(executable), str(ch_table), str(he_table)], timeout=60.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        shutil.rmtree(unit_build, ignore_errors=True)
