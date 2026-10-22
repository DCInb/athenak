cmake_minimum_required(VERSION 3.16)
project(thermal_cell_query LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT ATHENA_AUDIT_BUILD)
  message(FATAL_ERROR "Set ATHENA_AUDIT_BUILD to an existing Serial AthenaK build")
endif()
get_filename_component(ATHENA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
find_package(Kokkos CONFIG REQUIRED
  PATHS "${ATHENA_AUDIT_BUILD}/cmake_packages/Kokkos" NO_DEFAULT_PATH)
add_executable(thermal_cell_query thermal_cell_query.cpp
  "${ATHENA_ROOT}/src/materials/ionmix_two_temperature_table.cpp")
target_include_directories(thermal_cell_query PRIVATE
  "${ATHENA_ROOT}/src" "${ATHENA_AUDIT_BUILD}")
target_compile_definitions(thermal_cell_query PRIVATE KOKKOS_DEPENDENCE)
target_link_libraries(thermal_cell_query PRIVATE Kokkos::kokkos)
