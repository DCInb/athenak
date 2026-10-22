"""Device spatial lookup agrees with exhaustive ownership on irregular leaf meshes."""

from pathlib import Path
import shutil

import test_suite.testutils as testutils
from test_suite.nr.test_nr_biermann_endpoint_closure_cpu import cache_value


def test_block_lookup():
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    athena_build = Path.cwd().parent
    build = Path("laser_block_tree_unit_build")
    try:
        source = build / "source"
        source.mkdir(parents=True, exist_ok=True)
        (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.18)
project(laser_block_tree_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos CONFIG REQUIRED
  PATHS "{athena_build / 'cmake_packages/Kokkos'}" NO_DEFAULT_PATH)
add_executable(laser_block_tree_driver "{directory / 'laser_block_tree_driver.cpp'}")
target_include_directories(laser_block_tree_driver PRIVATE
  "{root / 'src'}" "{athena_build}")
target_compile_definitions(laser_block_tree_driver PRIVATE KOKKOS_DEPENDENCE)
target_link_libraries(laser_block_tree_driver PRIVATE Kokkos::kokkos)
""")
        compiler = cache_value(athena_build / "CMakeCache.txt", "CMAKE_CXX_COMPILER")
        assert testutils.run_command([
            "cmake", f"-DCMAKE_CXX_COMPILER={compiler}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-S", str(source), "-B", str(build / "build")])
        assert testutils.run_command(["cmake", "--build", str(build / "build"), "-j2"])
        assert testutils.run_command([str(build / "build/laser_block_tree_driver")],
                                     timeout=60)
    finally:
        shutil.rmtree(build, ignore_errors=True)
