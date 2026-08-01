"""CPU/GPU device regression for the shared Biermann endpoint closure."""

from pathlib import Path
import shutil

import pytest

import test_suite.testutils as testutils


TEST_DIRECTORY = Path(__file__).resolve().parent
SOURCE_ROOT = TEST_DIRECTORY.parents[2]
DRIVER = TEST_DIRECTORY / "biermann_endpoint_closure_driver.cpp"
UNIT_BUILD = Path("biermann_endpoint_closure_unit_build")


def cache_value(cache_path, key):
    prefix = f"{key}:"
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} is absent from {cache_path}")


def build_driver():
    athena_build = Path.cwd().parent
    kokkos_package = athena_build / "cmake_packages" / "Kokkos"
    cache_path = athena_build / "CMakeCache.txt"
    cmake_source = UNIT_BUILD / "source"
    cmake_source.mkdir(parents=True)
    cmake_text = f"""
cmake_minimum_required(VERSION 3.16)
project(biermann_endpoint_closure_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Kokkos CONFIG REQUIRED
             PATHS "{kokkos_package}" NO_DEFAULT_PATH)
add_executable(biermann_endpoint_closure_driver "{DRIVER}")
target_include_directories(biermann_endpoint_closure_driver PRIVATE
  "{SOURCE_ROOT / 'src'}" "{athena_build}")
target_compile_definitions(biermann_endpoint_closure_driver PRIVATE
  KOKKOS_DEPENDENCE)
target_link_libraries(biermann_endpoint_closure_driver PRIVATE Kokkos::kokkos)
"""
    (cmake_source / "CMakeLists.txt").write_text(cmake_text, encoding="ascii")
    compiler = cache_value(cache_path, "CMAKE_CXX_COMPILER")
    assert testutils.run_command([
        "cmake", f"-DCMAKE_CXX_COMPILER={compiler}",
        "-S", str(cmake_source), "-B", str(UNIT_BUILD / "build")])
    assert testutils.run_command([
        "cmake", "--build", str(UNIT_BUILD / "build"), "--parallel", "2"])
    return UNIT_BUILD / "build" / "biermann_endpoint_closure_driver"


def test_source_integration_uses_full_accepted_state():
    amr_source = (SOURCE_ROOT / "src/mhd/biermann_battery_amr.cpp").read_text(
        encoding="utf-8")
    task_source = (SOURCE_ROOT / "src/mhd/mhd_tasks.cpp").read_text(
        encoding="utf-8")
    assert "closure.CloseConserved(" in amr_source
    assert "coarse_b0" in amr_source
    assert "fine_magnetic" in amr_source
    close_position = task_source.index("&MHD::BiermannCloseInterior")
    restrict_position = task_source.index("&MHD::RestrictU", close_position)
    assert close_position < restrict_position


def test_shared_endpoint_closure_on_device():
    try:
        executable = build_driver()
        assert testutils.run_command([str(executable)], timeout=60.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        shutil.rmtree(UNIT_BUILD, ignore_errors=True)
