"""CGS stopping normalization, analytic range, and device angular-diffusion moments."""

import os
from pathlib import Path
import subprocess

import pytest


def test_fast_electron_physics(tmp_path):
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    requested = os.environ.get("ATHENAK_FAST_ELECTRONS_EXE")
    athena_build = (Path(requested).resolve().parents[1] if requested
                    else Path.cwd().parent)
    cache = athena_build / "CMakeCache.txt"
    if not cache.is_file():
        pytest.fail("Set ATHENAK_FAST_ELECTRONS_EXE to a CMake-built AthenaK executable")
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.16)
project(fast_electron_physics_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos CONFIG REQUIRED
  PATHS "{athena_build / 'cmake_packages/Kokkos'}" NO_DEFAULT_PATH)
add_executable(fast_electron_physics_driver
  "{directory / 'fast_electron_physics_driver.cpp'}")
target_include_directories(fast_electron_physics_driver PRIVATE
  "{root / 'src'}" "{athena_build}")
target_compile_definitions(fast_electron_physics_driver PRIVATE KOKKOS_DEPENDENCE)
target_link_libraries(fast_electron_physics_driver PRIVATE Kokkos::kokkos)
""", encoding="ascii")
    compiler = next(line.split("=", 1)[1] for line in cache.read_text().splitlines()
                    if line.startswith("CMAKE_CXX_COMPILER:"))
    build = tmp_path / "build"
    commands = [
        ["cmake", f"-DCMAKE_CXX_COMPILER={compiler}", "-DCMAKE_BUILD_TYPE=Release",
         "-S", str(source), "-B", str(build)],
        ["cmake", "--build", str(build), "-j2"],
        [str(build / "fast_electron_physics_driver")],
    ]
    for command in commands:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
        assert result.returncode == 0, result.stdout
