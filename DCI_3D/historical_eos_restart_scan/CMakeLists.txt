cmake_minimum_required(VERSION 3.16)
project(historical_eos_restart_scan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(ATHENAK_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ATHENAK_KOKKOS_DIR
    "${ATHENAK_SOURCE_DIR}/DCI_3D/build/cmake_packages/Kokkos"
    CACHE PATH "Kokkos package from the production AthenaK build")

find_package(Kokkos CONFIG REQUIRED PATHS "${ATHENAK_KOKKOS_DIR}" NO_DEFAULT_PATH)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

# The scanner is a single-process utility even when it links the CUDA Kokkos package
# used by the production executable.  Supply the small generated header expected by
# athena.hpp without inheriting MPI from DCI_3D/build/config.hpp.
set(PROBLEM historical_eos_restart_scan)
set(USER_PROBLEM_ENABLED 0)
set(TWO_PUNCTURES 0)
set(athena_VERSION_MAJOR 0)
set(athena_VERSION_MINOR 1)
set(SINGLE_PRECISION_ENABLED 0)
set(MPI_PARALLEL_ENABLED 0)
set(OPENMP_PARALLEL_ENABLED 0)
set(PAR_LOOP_LAYOUT "")
set(PAR_LOOP_INNER_LAYOUT "")
set(COMPILER_CHOICE "standalone scanner")
set(COMPILER_COMMAND "${CMAKE_CXX_COMPILER}")
set(COMPILER_FLAGS "${CMAKE_CXX_FLAGS}")
configure_file("${ATHENAK_SOURCE_DIR}/config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/config.hpp" @ONLY)

execute_process(
  COMMAND git -C "${ATHENAK_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE ATHENAK_SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
file(SHA256 "${ATHENAK_SOURCE_DIR}/src/materials/material_mixture.hpp"
     MATERIAL_MIXTURE_SHA256)
file(SHA256 "${ATHENAK_SOURCE_DIR}/src/materials/ionmix_two_temperature_table.hpp"
     IONMIX_HEADER_SHA256)
file(SHA256 "${ATHENAK_SOURCE_DIR}/src/materials/ionmix_two_temperature_table.cpp"
     IONMIX_SOURCE_SHA256)
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/historical_eos_restart_scan.cpp"
     SCANNER_SOURCE_SHA256)

add_executable(historical_eos_restart_scan
  historical_eos_restart_scan.cpp
  "${ATHENAK_SOURCE_DIR}/src/materials/ionmix_two_temperature_table.cpp")
target_include_directories(historical_eos_restart_scan PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${ATHENAK_SOURCE_DIR}/src")
target_compile_definitions(historical_eos_restart_scan PRIVATE
  KOKKOS_DEPENDENCE
  ATHENAK_SOURCE_REVISION="${ATHENAK_SOURCE_REVISION}"
  MATERIAL_MIXTURE_SHA256="${MATERIAL_MIXTURE_SHA256}"
  IONMIX_HEADER_SHA256="${IONMIX_HEADER_SHA256}"
  IONMIX_SOURCE_SHA256="${IONMIX_SOURCE_SHA256}"
  SCANNER_SOURCE_SHA256="${SCANNER_SOURCE_SHA256}")
target_link_libraries(historical_eos_restart_scan PRIVATE
  Kokkos::kokkos
  OpenSSL::Crypto)
