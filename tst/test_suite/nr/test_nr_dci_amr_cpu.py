"""Build the material refinement checks against an existing AthenaK Kokkos build."""

import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]


def test_dci_amr_material_policy(tmp_path):
    build = Path(os.environ.get("ATHENA_DCI_TEST_BUILD", Path.cwd().parent)).resolve()
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.16)
project(dci_amr_checks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos CONFIG REQUIRED PATHS "{build}/cmake_packages/Kokkos"
             NO_DEFAULT_PATH)
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(dci_amr_driver "{REPO}/tst/test_suite/nr/dci_amr_driver.cpp")
target_include_directories(dci_amr_driver PRIVATE "{REPO}" "{REPO}/src" "{build}")
target_link_libraries(dci_amr_driver PRIVATE Kokkos::kokkos MPI::MPI_CXX)
""", encoding="ascii")
    commands = [
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        ["cmake", "--build", str(tmp_path / "build"), "--parallel", "2"],
        [str(tmp_path / "build/dci_amr_driver")],
    ]
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout)
