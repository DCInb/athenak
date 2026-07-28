"""Standalone native separate-ion/electron IONMIX table regression test."""

from pathlib import Path
import shutil

import pytest

import test_suite.testutils as testutils


test_directory = Path(__file__).resolve().parent
source_root = test_directory.parents[2]
driver_source = test_directory / "ionmix_two_temperature_table_driver.cpp"
fixture = test_directory / "ionmix_two_temperature_table_fixture.dat"
unit_build = Path("ionmix_two_temperature_table_unit_build")


def configure_standalone_test():
    """Compile the new source without changing AthenaK's integration CMake list."""
    athena_build = Path.cwd().parent
    kokkos_package = athena_build / "cmake_packages" / "Kokkos"
    assert (kokkos_package / "KokkosConfig.cmake").is_file()
    cmake_source = unit_build / "source"
    cmake_source.mkdir(parents=True)
    cmake_text = f"""
cmake_minimum_required(VERSION 3.16)
project(ionmix_two_temperature_table_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Kokkos CONFIG REQUIRED
             PATHS \"{kokkos_package}\" NO_DEFAULT_PATH)
add_executable(ionmix_two_temperature_table_driver
  \"{driver_source}\"
  \"{source_root / 'src/materials/ionmix_two_temperature_table.cpp'}\")
target_include_directories(ionmix_two_temperature_table_driver PRIVATE
  \"{source_root / 'src'}\" \"{athena_build}\")
target_compile_definitions(ionmix_two_temperature_table_driver PRIVATE
  KOKKOS_DEPENDENCE)
target_link_libraries(ionmix_two_temperature_table_driver PRIVATE Kokkos::kokkos)
"""
    (cmake_source / "CMakeLists.txt").write_text(cmake_text, encoding="ascii")
    assert testutils.run_command([
        "cmake", "-S", str(cmake_source), "-B", str(unit_build / "build")])
    assert testutils.run_command([
        "cmake", "--build", str(unit_build / "build"), "--parallel", "2"])
    return unit_build / "build" / "ionmix_two_temperature_table_driver"


def test_run():
    try:
        executable = configure_standalone_test()
        assert testutils.run_command([
            str(executable), "check", str(fixture)], timeout=60.0)

        # Error bounds must stop a device query instead of silently clamping it.
        assert not testutils.run_command([
            str(executable), "error_bounds", str(fixture)], timeout=60.0)

        # A decreasing component-energy row is rejected at load time.  The fixture is
        # copied rather than edited so its exact-file fingerprint remains testable.
        invalid = unit_build / "decreasing_energy.dat"
        contents = fixture.read_text(encoding="ascii")
        assert contents.count("10.0 10.0 80.0") == 1
        invalid.write_text(
            contents.replace("10.0 10.0 80.0", "10.0 9.0 80.0"),
            encoding="ascii")
        assert not testutils.run_command([
            str(executable), "load_only", str(invalid)], timeout=60.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        shutil.rmtree(unit_build, ignore_errors=True)
