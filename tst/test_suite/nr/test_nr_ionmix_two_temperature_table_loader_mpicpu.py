"""MPI rank-zero-read regression test for the separate-ion/electron table."""

import os
from pathlib import Path
import shutil
import threading

import pytest

import test_suite.testutils as testutils


test_directory = Path(__file__).resolve().parent
source_root = test_directory.parents[2]
driver_source = test_directory / "ionmix_two_temperature_table_driver.cpp"
fixture = test_directory / "ionmix_two_temperature_table_fixture.dat"
unit_build = Path("ionmix_two_temperature_table_mpi_unit_build")
fifo_table = Path("ionmix_two_temperature_rank0.fifo")


def cmake_cache_path(cache, key):
    """Return one FILEPATH entry from the parent AthenaK configuration."""
    prefix = f"{key}:FILEPATH="
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix):])
    raise AssertionError(f"{key} is missing from {cache}")


def configure_standalone_mpi_test():
    """Build against the MPI configuration selected by the AthenaK test runner."""
    athena_build = Path.cwd().parent
    kokkos_package = athena_build / "cmake_packages" / "Kokkos"
    assert (kokkos_package / "KokkosConfig.cmake").is_file()
    mpi_compiler = cmake_cache_path(
        athena_build / "CMakeCache.txt", "MPI_CXX_COMPILER")
    assert mpi_compiler.is_file()
    cmake_source = unit_build / "source"
    cmake_source.mkdir(parents=True)
    cmake_text = f"""
cmake_minimum_required(VERSION 3.16)
project(ionmix_two_temperature_table_mpi_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Kokkos CONFIG REQUIRED
             PATHS \"{kokkos_package}\" NO_DEFAULT_PATH)
add_executable(ionmix_two_temperature_table_driver
  \"{driver_source}\"
  \"{source_root / 'src/materials/ionmix_two_temperature_table.cpp'}\")
target_include_directories(ionmix_two_temperature_table_driver PRIVATE
  \"{source_root / 'src'}\" \"{athena_build}\")
target_compile_definitions(ionmix_two_temperature_table_driver PRIVATE
  KOKKOS_DEPENDENCE)
target_link_libraries(ionmix_two_temperature_table_driver PRIVATE
  Kokkos::kokkos MPI::MPI_CXX)
"""
    (cmake_source / "CMakeLists.txt").write_text(cmake_text, encoding="ascii")
    assert testutils.run_command([
        "cmake", f"-DMPI_CXX_COMPILER={mpi_compiler}",
        "-S", str(cmake_source), "-B", str(unit_build / "build")])
    assert testutils.run_command([
        "cmake", "--build", str(unit_build / "build"), "--parallel", "2"])
    return unit_build / "build" / "ionmix_two_temperature_table_driver"


def test_run():
    writer = None
    writer_errors = []
    try:
        executable = configure_standalone_mpi_test()
        mpiexec = cmake_cache_path(
            Path.cwd().parent / "CMakeCache.txt", "MPIEXEC_EXECUTABLE")
        assert mpiexec.is_file()
        table_text = fixture.read_text(encoding="ascii")
        fifo_table.unlink(missing_ok=True)
        os.mkfifo(fifo_table)

        def write_table_once():
            try:
                with fifo_table.open("w", encoding="ascii") as stream:
                    stream.write(table_text)
            except Exception as exc:  # pragma: no cover - reported below
                writer_errors.append(exc)

        writer = threading.Thread(target=write_table_once, daemon=True)
        writer.start()
        assert testutils.run_command([
            str(mpiexec), "-np", "2", str(executable),
            "mpi_check", str(fifo_table)], timeout=60.0)
        writer.join(timeout=10.0)
        assert not writer.is_alive(), "FIFO table writer did not finish."
        assert not writer_errors, str(writer_errors[0]) if writer_errors else ""
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        if writer is not None:
            writer.join(timeout=1.0)
        fifo_table.unlink(missing_ok=True)
        shutil.rmtree(unit_build, ignore_errors=True)
