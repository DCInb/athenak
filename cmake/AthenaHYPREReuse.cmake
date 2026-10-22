# Optional HYPRE 2.33 CUDA transfer reuse. The private setup has an Athena
# symbol name; ordinary HYPRE users continue calling the installed library.
if (Athena_ENABLE_HYPRE_REUSE)
  if (NOT Athena_ENABLE_HYPRE OR NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "Athena_ENABLE_HYPRE_REUSE requires Athena_ENABLE_HYPRE "
        "and the qualified CUDA backend")
  endif()
  get_target_property(_athena_hypre_type HYPRE::HYPRE TYPE)
  if (NOT _athena_hypre_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "Athena_ENABLE_HYPRE_REUSE requires the qualified static HYPRE build")
  endif()
  unset(_athena_hypre_type)
  # The source also checks HYPRE_RELEASE_NUMBER == 23300 at compile time.
  add_library(athena_hypre_reuse OBJECT
      ${CMAKE_CURRENT_SOURCE_DIR}/src/two_temperature/hypre_233_reuse.cpp)
  target_link_libraries(athena_hypre_reuse PRIVATE Kokkos::kokkos HYPRE::HYPRE)
  if (ENABLE_MPI)
    target_link_libraries(athena_hypre_reuse PRIVATE MPI::MPI_CXX)
  endif()
  foreach(_athena_target athena thermal_radiation_hypre_driver thermal_radiation_hypre_mpi_driver)
    if (TARGET ${_athena_target})
      target_sources(${_athena_target} PRIVATE $<TARGET_OBJECTS:athena_hypre_reuse>)
      if (NOT _athena_target STREQUAL "athena")
        target_compile_definitions(${_athena_target} PRIVATE ATHENA_HYPRE_REUSE_ENABLED=1)
      endif()
    endif()
  endforeach()
  unset(_athena_target)
endif()
