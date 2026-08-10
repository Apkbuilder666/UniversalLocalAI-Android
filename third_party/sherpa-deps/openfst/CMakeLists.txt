cmake_minimum_required(VERSION 3.13)

project(openfst VERSION 1.8.5 LANGUAGES CXX)

include(CMakeParseArguments)
include(GNUInstallDirs)

option(BUILD_TESTING "Build OpenFst tests" OFF)
include(CTest)

if(WIN32)
  option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
  if(MSVC)
    add_compile_options(/bigobj)
  endif()
else()
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
endif()

option(HAVE_BIN "Build fst::script and command-line binaries" ON)
option(HAVE_SCRIPT "Build the fst::script library" ON)
option(HAVE_COMPACT "Build compact FST extensions" ON)
option(HAVE_COMPRESS "Build compress extension" ON)
option(HAVE_CONST "Build const FST extensions" ON)
option(HAVE_FAR "Build FAR extensions" ON)
option(HAVE_FSTS "Enable all FST shared-object extensions" OFF)
option(HAVE_GRM "Enable FAR, PDT, and MPDT extensions" OFF)
option(HAVE_LINEAR "Build linear FST extensions" ON)
option(HAVE_LOOKAHEAD "Build lookahead FST extensions" ON)
option(HAVE_MPDT "Build MPDT extensions" ON)
option(HAVE_NGRAM "Build n-gram FST extensions" ON)
option(HAVE_PDT "Build PDT extensions" ON)
option(HAVE_SPECIAL "Build special matcher extensions" ON)
option(OPENFST_USE_ABSL "Use Abseil synchronization primitives" OFF)


if(HAVE_FSTS)
  set(HAVE_COMPACT ON CACHE BOOL "" FORCE)
  set(HAVE_CONST ON CACHE BOOL "" FORCE)
  set(HAVE_LINEAR ON CACHE BOOL "" FORCE)
  set(HAVE_LOOKAHEAD ON CACHE BOOL "" FORCE)
  set(HAVE_NGRAM ON CACHE BOOL "" FORCE)
endif()

if(HAVE_GRM)
  set(HAVE_FAR ON CACHE BOOL "" FORCE)
  set(HAVE_MPDT ON CACHE BOOL "" FORCE)
  set(HAVE_PDT ON CACHE BOOL "" FORCE)
endif()

if(HAVE_BIN)
  set(HAVE_SCRIPT ON CACHE BOOL "" FORCE)
endif()

if(HAVE_MPDT)
  set(HAVE_PDT ON CACHE BOOL "" FORCE)
endif()

message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "HAVE_BIN: ${HAVE_BIN}")
message(STATUS "HAVE_SCRIPT: ${HAVE_SCRIPT}")
message(STATUS "HAVE_COMPACT: ${HAVE_COMPACT}")
message(STATUS "HAVE_COMPRESS: ${HAVE_COMPRESS}")
message(STATUS "HAVE_CONST: ${HAVE_CONST}")
message(STATUS "HAVE_FAR: ${HAVE_FAR}")
message(STATUS "HAVE_FSTS: ${HAVE_FSTS}")
message(STATUS "HAVE_GRM: ${HAVE_GRM}")
message(STATUS "HAVE_LINEAR: ${HAVE_LINEAR}")
message(STATUS "HAVE_LOOKAHEAD: ${HAVE_LOOKAHEAD}")
message(STATUS "HAVE_MPDT: ${HAVE_MPDT}")
message(STATUS "HAVE_NGRAM: ${HAVE_NGRAM}")
message(STATUS "HAVE_PDT: ${HAVE_PDT}")
message(STATUS "HAVE_SPECIAL: ${HAVE_SPECIAL}")
message(STATUS "OPENFST_USE_ABSL: ${OPENFST_USE_ABSL}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MACOSX_RPATH ON)

set(OPENFST_SOVERSION 27)
set(OPENFST_LIBRARY_VERSION "${OPENFST_SOVERSION}.0.0")
set(OPENFST_MODULE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/fst" CACHE PATH
    "Install directory for dynamically loaded FST modules")

set(OPENFST_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/include")
set(OPENFST_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(WIN32 AND MINGW)
  # MinGW's <cmath> does not expose C math constants like M_LN2 by default.
  add_compile_definitions(M_LN2=0.69314718055994530942)
endif()

set(OPENFST_SYSTEM_LIBS "")
if(NOT WIN32 OR MINGW)
  list(APPEND OPENFST_SYSTEM_LIBS m)
endif()
if(CMAKE_DL_LIBS)
  list(APPEND OPENFST_SYSTEM_LIBS ${CMAKE_DL_LIBS})
endif()

if(OPENFST_USE_ABSL)
  find_package(absl CONFIG REQUIRED COMPONENTS synchronization)
endif()

function(openfst_configure_target target)
  get_target_property(target_type ${target} TYPE)
  if(target_type STREQUAL "EXECUTABLE")
    target_include_directories(${target} PRIVATE "${OPENFST_PUBLIC_INCLUDE_DIR}")
  else()
    target_include_directories(
        ${target}
        PUBLIC
          "$<BUILD_INTERFACE:${OPENFST_PUBLIC_INCLUDE_DIR}>"
          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  endif()

  target_compile_features(${target} PUBLIC cxx_std_17)

  if(WIN32)
    target_compile_definitions(${target} PUBLIC FST_NO_DYNAMIC_LINKING)
    if(MSVC)
      target_compile_options(${target} PRIVATE /bigobj)
    else()
      target_compile_options(
          ${target}
          PRIVATE
            -fno-exceptions
            -Wno-deprecated-declarations
            -Wno-sign-compare)
    endif()
  else()
    target_compile_options(
        ${target}
        PRIVATE
          -fno-exceptions
          -Wno-deprecated-declarations
          -Wno-sign-compare)
  endif()
endfunction()

function(openfst_add_library target)
  set(options)
  set(one_value_args OUTPUT_NAME)
  set(multi_value_args SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_library(${target} ${ARG_SOURCES})
  openfst_configure_target(${target})

  if(ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()
  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${target} PUBLIC ${ARG_LINK_LIBRARIES})
  endif()
  if(ARG_OUTPUT_NAME)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${ARG_OUTPUT_NAME}")
  endif()

  set_target_properties(
      ${target}
      PROPERTIES
        VERSION "${OPENFST_LIBRARY_VERSION}"
        SOVERSION "${OPENFST_SOVERSION}")

  install(
      TARGETS ${target}
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()

function(openfst_add_module target)
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_library(${target} MODULE ${ARG_SOURCES})
  openfst_configure_target(${target})

  if(ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()
  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${target} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()

  set_target_properties(${target} PROPERTIES PREFIX "")

  install(
      TARGETS ${target}
      LIBRARY DESTINATION "${OPENFST_MODULE_INSTALL_DIR}"
      ARCHIVE DESTINATION "${OPENFST_MODULE_INSTALL_DIR}"
      RUNTIME DESTINATION "${OPENFST_MODULE_INSTALL_DIR}")
endfunction()

function(openfst_add_binary target)
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_executable(${target} ${ARG_SOURCES})
  openfst_configure_target(${target})

  if(ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()
  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${target} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()

  install(TARGETS ${target} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()

set(OPENFST_CORE_SOURCES
    src/lib/compat.cc
    src/lib/encode.cc
    src/lib/flags.cc
    src/lib/fst.cc
    src/lib/fst-types.cc
    src/lib/mapped-file.cc
    src/lib/properties.cc
    src/lib/symbol-table.cc
    src/lib/symbol-table-ops.cc
    src/lib/util.cc
    src/lib/weight.cc)

openfst_add_library(
    fst
    SOURCES ${OPENFST_CORE_SOURCES}
    LINK_LIBRARIES ${OPENFST_SYSTEM_LIBS})

if(OPENFST_USE_ABSL)
  target_compile_definitions(fst PUBLIC OPENFST_HAS_ABSL=1)
  target_link_libraries(fst PUBLIC absl::synchronization)
endif()

if(HAVE_SCRIPT)
  set(OPENFST_SCRIPT_SOURCES
      src/script/arciterator-class.cc
      src/script/arcsort.cc
      src/script/closure.cc
      src/script/compile.cc
      src/script/compose.cc
      src/script/concat.cc
      src/script/connect.cc
      src/script/convert.cc
      src/script/decode.cc
      src/script/determinize.cc
      src/script/difference.cc
      src/script/disambiguate.cc
      src/script/draw.cc
      src/script/encode.cc
      src/script/encodemapper-class.cc
      src/script/epsnormalize.cc
      src/script/equal.cc
      src/script/equivalent.cc
      src/script/fst-class.cc
      src/script/getters.cc
      src/script/info-impl.cc
      src/script/info.cc
      src/script/intersect.cc
      src/script/invert.cc
      src/script/isomorphic.cc
      src/script/map.cc
      src/script/minimize.cc
      src/script/print.cc
      src/script/project.cc
      src/script/prune.cc
      src/script/push.cc
      src/script/randequivalent.cc
      src/script/randgen.cc
      src/script/relabel.cc
      src/script/replace.cc
      src/script/reverse.cc
      src/script/reweight.cc
      src/script/rmepsilon.cc
      src/script/shortest-distance.cc
      src/script/shortest-path.cc
      src/script/stateiterator-class.cc
      src/script/synchronize.cc
      src/script/text-io.cc
      src/script/topsort.cc
      src/script/union.cc
      src/script/weight-class.cc
      src/script/verify.cc)

  openfst_add_library(
      fstscript
      SOURCES ${OPENFST_SCRIPT_SOURCES}
      LINK_LIBRARIES fst)

  if(MSVC)
    target_compile_options(
        fstscript
        PRIVATE
          $<$<CONFIG:Release>:/Od /Ob0>
          $<$<CONFIG:RelWithDebInfo>:/Od /Ob0>
          $<$<CONFIG:MinSizeRel>:/Od /Ob0>)
  endif()
endif()

if(HAVE_BIN)
  set(OPENFST_BINARIES
      fstarcsort
      fstclosure
      fstcompile
      fstcompose
      fstconcat
      fstconnect
      fstconvert
      fstdeterminize
      fstdifference
      fstdisambiguate
      fstdraw
      fstencode
      fstepsnormalize
      fstequal
      fstequivalent
      fstinfo
      fstintersect
      fstinvert
      fstisomorphic
      fstmap
      fstminimize
      fstprint
      fstproject
      fstprune
      fstpush
      fstrandgen
      fstrelabel
      fstreplace
      fstreverse
      fstreweight
      fstrmepsilon
      fstshortestdistance
      fstshortestpath
      fstsymbols
      fstsynchronize
      fsttopsort
      fstunion)

  foreach(binary IN LISTS OPENFST_BINARIES)
    openfst_add_binary(
        ${binary}
        SOURCES "src/bin/${binary}.cc" "src/bin/${binary}-main.cc"
        LINK_LIBRARIES fstscript fst)
  endforeach()
endif()

if(HAVE_COMPACT)
  set(OPENFST_COMPACT_MODULES
      compact8_acceptor-fst
      compact8_string-fst
      compact8_unweighted-fst
      compact8_unweighted_acceptor-fst
      compact8_weighted_string-fst
      compact16_acceptor-fst
      compact16_string-fst
      compact16_unweighted-fst
      compact16_unweighted_acceptor-fst
      compact16_weighted_string-fst
      compact64_acceptor-fst
      compact64_string-fst
      compact64_unweighted-fst
      compact64_unweighted_acceptor-fst
      compact64_weighted_string-fst)

  set(OPENFST_COMPACT_SOURCES "")
  foreach(module IN LISTS OPENFST_COMPACT_MODULES)
    list(APPEND OPENFST_COMPACT_SOURCES "src/extensions/compact/${module}.cc")
  endforeach()

  openfst_add_library(
      fstcompact
      SOURCES ${OPENFST_COMPACT_SOURCES}
      LINK_LIBRARIES fst)

  foreach(module IN LISTS OPENFST_COMPACT_MODULES)
    openfst_add_module(
        ${module}
        SOURCES "src/extensions/compact/${module}.cc"
        LINK_LIBRARIES fst)
  endforeach()
endif()

if(HAVE_COMPRESS AND HAVE_SCRIPT)
  openfst_add_library(
      fstcompressscript
      SOURCES src/extensions/compress/compressscript.cc
      LINK_LIBRARIES fstscript fst)

  if(HAVE_BIN)
    openfst_add_binary(
        fstcompress
        SOURCES
          src/extensions/compress/fstcompress.cc
          src/extensions/compress/fstcompress-main.cc
        LINK_LIBRARIES fstcompressscript fstscript fst)
  endif()
endif()

if(HAVE_CONST)
  set(OPENFST_CONST_MODULES const8-fst const16-fst const64-fst)
  set(OPENFST_CONST_SOURCES "")
  foreach(module IN LISTS OPENFST_CONST_MODULES)
    list(APPEND OPENFST_CONST_SOURCES "src/extensions/const/${module}.cc")
  endforeach()

  openfst_add_library(
      fstconst
      SOURCES ${OPENFST_CONST_SOURCES}
      LINK_LIBRARIES fst)

  foreach(module IN LISTS OPENFST_CONST_MODULES)
    openfst_add_module(
        ${module}
        SOURCES "src/extensions/const/${module}.cc"
        LINK_LIBRARIES fst)
  endforeach()
endif()

if(HAVE_FAR)
  openfst_add_library(
      fstfar
      SOURCES
        src/extensions/far/far.cc
        src/extensions/far/stlist.cc
        src/extensions/far/sttable.cc
      LINK_LIBRARIES fst)

  if(HAVE_SCRIPT)
    openfst_add_library(
        fstfarscript
        SOURCES
          src/extensions/far/compile-strings.cc
          src/extensions/far/far-class.cc
          src/extensions/far/farscript.cc
          src/extensions/far/getters.cc
          src/extensions/far/script-impl.cc
        LINK_LIBRARIES fstfar fstscript fst)

    if(HAVE_BIN)
      set(OPENFST_FAR_BINARIES
          farcompilestrings
          farconvert
          farcreate
          farencode
          farequal
          farextract
          farinfo
          farisomorphic
          farprintstrings)

      foreach(binary IN LISTS OPENFST_FAR_BINARIES)
        openfst_add_binary(
            ${binary}
            SOURCES
              "src/extensions/far/${binary}.cc"
              "src/extensions/far/${binary}-main.cc"
            LINK_LIBRARIES fstfarscript fstscript fstfar fst)
      endforeach()
    endif()
  endif()
endif()

if(HAVE_LINEAR)
  openfst_add_library(
      openfstlinear
      OUTPUT_NAME fstlinear
      SOURCES
        src/extensions/linear/linear-classifier-fst.cc
        src/extensions/linear/linear-tagger-fst.cc
      LINK_LIBRARIES fst)

  foreach(module IN ITEMS linear-classifier-fst linear-tagger-fst)
    openfst_add_module(
        ${module}
        SOURCES "src/extensions/linear/${module}.cc"
        LINK_LIBRARIES openfstlinear fst)
  endforeach()

  if(HAVE_SCRIPT)
    openfst_add_library(
        fstlinearscript
        SOURCES src/extensions/linear/linearscript.cc
        LINK_LIBRARIES openfstlinear fstscript fst)

    if(HAVE_BIN)
      foreach(binary IN ITEMS fstlinear fstloglinearapply)
        openfst_add_binary(
            ${binary}
            SOURCES
              "src/extensions/linear/${binary}.cc"
              "src/extensions/linear/${binary}-main.cc"
            LINK_LIBRARIES fstlinearscript openfstlinear fstscript fst)
      endforeach()
    endif()
  endif()
endif()

if(HAVE_LOOKAHEAD)
  set(OPENFST_LOOKAHEAD_MODULES
      arc_lookahead-fst
      ilabel_lookahead-fst
      olabel_lookahead-fst)
  set(OPENFST_LOOKAHEAD_SOURCES "")
  foreach(module IN LISTS OPENFST_LOOKAHEAD_MODULES)
    list(APPEND OPENFST_LOOKAHEAD_SOURCES "src/extensions/lookahead/${module}.cc")
  endforeach()

  openfst_add_library(
      fstlookahead
      SOURCES ${OPENFST_LOOKAHEAD_SOURCES}
      LINK_LIBRARIES fst)

  foreach(module IN LISTS OPENFST_LOOKAHEAD_MODULES)
    openfst_add_module(
        ${module}
        SOURCES "src/extensions/lookahead/${module}.cc"
        LINK_LIBRARIES fst)
  endforeach()
endif()

if(HAVE_PDT AND HAVE_SCRIPT)
  openfst_add_library(
      fstpdtscript
      SOURCES
        src/extensions/pdt/getters.cc
        src/extensions/pdt/pdtscript.cc
      LINK_LIBRARIES fstscript fst)

  if(HAVE_BIN)
    set(OPENFST_PDT_BINARIES
        pdtcompose
        pdtexpand
        pdtinfo
        pdtreplace
        pdtreverse
        pdtshortestpath)

    foreach(binary IN LISTS OPENFST_PDT_BINARIES)
      openfst_add_binary(
          ${binary}
          SOURCES
            "src/extensions/pdt/${binary}.cc"
            "src/extensions/pdt/${binary}-main.cc"
          LINK_LIBRARIES fstpdtscript fstscript fst)
    endforeach()
  endif()
endif()

if(HAVE_MPDT AND HAVE_SCRIPT)
  openfst_add_library(
      fstmpdtscript
      SOURCES src/extensions/mpdt/mpdtscript.cc
      LINK_LIBRARIES fstpdtscript fstscript fst)

  if(HAVE_BIN)
    set(OPENFST_MPDT_BINARIES
        mpdtcompose
        mpdtexpand
        mpdtinfo
        mpdtreverse)

    foreach(binary IN LISTS OPENFST_MPDT_BINARIES)
      openfst_add_binary(
          ${binary}
          SOURCES
            "src/extensions/mpdt/${binary}.cc"
            "src/extensions/mpdt/${binary}-main.cc"
          LINK_LIBRARIES fstmpdtscript fstpdtscript fstscript fst)
    endforeach()
  endif()
endif()

if(HAVE_NGRAM)
  set(OPENFST_NGRAM_SOURCES
      src/extensions/ngram/bitmap-index.cc
      src/extensions/ngram/ngram-fst.cc
      src/extensions/ngram/nthbit.cc)

  openfst_add_library(
      fstngram
      SOURCES ${OPENFST_NGRAM_SOURCES}
      LINK_LIBRARIES fst)

  openfst_add_module(
      ngram-fst
      SOURCES ${OPENFST_NGRAM_SOURCES}
      LINK_LIBRARIES fst)
endif()

if(HAVE_SPECIAL)
  set(OPENFST_SPECIAL_SOURCES
      src/extensions/special/phi-fst.cc
      src/extensions/special/rho-fst.cc
      src/extensions/special/sigma-fst.cc)

  openfst_add_library(
      openfstspecial
      OUTPUT_NAME fstspecial
      SOURCES ${OPENFST_SPECIAL_SOURCES}
      LINK_LIBRARIES fst)

  foreach(module IN ITEMS phi-fst rho-fst sigma-fst)
    openfst_add_module(
        ${module}
        SOURCES "src/extensions/special/${module}.cc"
        LINK_LIBRARIES fst)
  endforeach()

  if(HAVE_BIN AND HAVE_SCRIPT)
    openfst_add_binary(
        fstspecial
        SOURCES src/extensions/special/fstspecial.cc
        INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src/bin"
        LINK_LIBRARIES openfstspecial fstscript fst)
  endif()
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/include/fst"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/COPYING"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/openfst")

if(BUILD_TESTING)
  openfst_add_binary(
      fst_test
      SOURCES src/test/fst_test.cc
      LINK_LIBRARIES fst)
  add_test(NAME fst_test COMMAND fst_test)

  openfst_add_binary(
      weight_test
      SOURCES src/test/weight_test.cc
      LINK_LIBRARIES fst)
  add_test(NAME weight_test COMMAND weight_test)

  foreach(weight IN ITEMS log tropical minmax lexicographic power)
    string(TOUPPER "${weight}" weight_upper)
    set(test_name "algo_test_${weight}")
    add_executable(${test_name} src/test/algo_test.cc)
    openfst_configure_target(${test_name})
    target_compile_definitions(${test_name} PRIVATE "TEST_${weight_upper}")
    target_link_libraries(${test_name} PRIVATE fst)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
