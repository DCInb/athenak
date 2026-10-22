# Optional SuiteSparse host factorization/solve dependency. No site-specific paths.
set(CHOLMOD_ROOT "" CACHE PATH "SuiteSparse installation prefix")
option(CHOLMOD_USE_STATIC_LIBS "Prefer static CHOLMOD and ordering libraries" OFF)
set(CHOLMOD_EXTRA_LIBRARIES "" CACHE STRING "Additional CHOLMOD link dependencies")
find_path(CHOLMOD_INCLUDE_DIR cholmod.h HINTS ${CHOLMOD_ROOT}
    PATH_SUFFIXES include/suitesparse include)
foreach(component cholmod amd colamd camd ccolamd suitesparseconfig)
  string(TOUPPER "${component}" component_upper)
  set(component_names ${component})
  if (CHOLMOD_USE_STATIC_LIBS)
    list(INSERT component_names 0
        "${CMAKE_STATIC_LIBRARY_PREFIX}${component}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  endif()
  find_library(CHOLMOD_${component_upper}_LIBRARY NAMES ${component_names}
      HINTS ${CHOLMOD_ROOT} PATH_SUFFIXES lib lib64)
  if (NOT CHOLMOD_${component_upper}_LIBRARY)
    message(FATAL_ERROR "Athena_ENABLE_CHOLMOD requires SuiteSparse ${component}. "
        "Set CHOLMOD_ROOT or CHOLMOD_${component_upper}_LIBRARY.")
  endif()
  list(APPEND CHOLMOD_LINK_LIBRARIES ${CHOLMOD_${component_upper}_LIBRARY})
endforeach()
if (NOT CHOLMOD_INCLUDE_DIR)
  message(FATAL_ERROR "Athena_ENABLE_CHOLMOD requires cholmod.h. "
      "Set CHOLMOD_ROOT or CHOLMOD_INCLUDE_DIR.")
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# Some static SuiteSparse packages use OpenMP internally. Link its runtime without
# changing Athena/Kokkos compilation options or enabling OpenMP kernels.
find_package(OpenMP QUIET COMPONENTS CXX)
list(APPEND CHOLMOD_LINK_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES} ${CHOLMOD_EXTRA_LIBRARIES})

# Catch incomplete shared installations and missing static transitive dependencies
# at configuration time, before a lengthy CUDA build. Exercise the factor/solve
# symbols as well as initialization, so the BLAS/LAPACK dependencies are checked.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${CHOLMOD_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${CHOLMOD_LINK_LIBRARIES})
unset(CHOLMOD_LINKS CACHE)
check_cxx_source_compiles("
#include <cholmod.h>
int main() {
  cholmod_common c;
  cholmod_start(&c);
  cholmod_sparse *a = cholmod_allocate_sparse(1, 1, 1, 1, 1, -1, CHOLMOD_REAL, &c);
  static_cast<int *>(a->p)[0] = 0;
  static_cast<int *>(a->p)[1] = 1;
  static_cast<int *>(a->i)[0] = 0;
  static_cast<double *>(a->x)[0] = 1;
  cholmod_factor *f = cholmod_analyze(a, &c);
  cholmod_factorize(a, f, &c);
  cholmod_dense *b = cholmod_ones(1, 1, CHOLMOD_REAL, &c);
  cholmod_dense *x = nullptr, *y = nullptr, *e = nullptr;
  cholmod_solve2(CHOLMOD_A, f, b, nullptr, &x, nullptr, &y, &e, &c);
  cholmod_factor *copy = cholmod_copy_factor(f, &c);
  cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, copy, &c);
  cholmod_sparse *l = cholmod_factor_to_sparse(copy, &c);
  cholmod_free_sparse(&l, &c);
  cholmod_free_factor(&copy, &c);
  cholmod_free_dense(&e, &c);
  cholmod_free_dense(&y, &c);
  cholmod_free_dense(&x, &c);
  cholmod_free_dense(&b, &c);
  cholmod_free_factor(&f, &c);
  cholmod_free_sparse(&a, &c);
  return !cholmod_finish(&c);
}" CHOLMOD_LINKS)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if (NOT CHOLMOD_LINKS)
  message(FATAL_ERROR "Athena_ENABLE_CHOLMOD: SuiteSparse factor/solve link check failed. "
      "Check its BLAS/LAPACK and runtime dependencies, or select CHOLMOD_USE_STATIC_LIBS "
      "and provide CHOLMOD_EXTRA_LIBRARIES if required. See CMakeFiles/CMakeError.log.")
endif()
