#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clblast" for configuration ""
set_property(TARGET clblast APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(clblast PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libclblast.so.1.6.3"
  IMPORTED_SONAME_NOCONFIG "libclblast.so.1"
  )

list(APPEND _cmake_import_check_targets clblast )
list(APPEND _cmake_import_check_files_for_clblast "${_IMPORT_PREFIX}/lib/libclblast.so.1.6.3" )

# Import target "tart" for configuration ""
set_property(TARGET tart APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(tart PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_NOCONFIG "subprocess"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtart.so"
  IMPORTED_SONAME_NOCONFIG "libtart.so"
  )

list(APPEND _cmake_import_check_targets tart )
list(APPEND _cmake_import_check_files_for_tart "${_IMPORT_PREFIX}/lib/libtart.so" )

# Import target "subprocess" for configuration ""
set_property(TARGET subprocess APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(subprocess PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsubprocess.so"
  IMPORTED_SONAME_NOCONFIG "libsubprocess.so"
  )

list(APPEND _cmake_import_check_targets subprocess )
list(APPEND _cmake_import_check_files_for_subprocess "${_IMPORT_PREFIX}/lib/libsubprocess.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
