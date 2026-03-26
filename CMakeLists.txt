cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

########################################################################
####                        Project metadata                        ####
########################################################################

project(SAMURAI-KOKKOS-Expermients
	VERSION 1.0.0
	LANGUAGES CXX)

########################################################################
####                           Variables                            ####
########################################################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

set(SAMURAI_KOKKOS_SAN_FLAGS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)

set(SAMURAI_KOKKOS_COMMON_COMPILE_OPTIONS
	-Wall
	-Wcast-align
	-Wconversion
	-Wdouble-promotion
	-Werror
	-Wextra
	-Wformat=2
	-Winit-self
	-Wmissing-declarations
	-Wno-error=array-bounds
	-Wno-error=conversion
	-Wno-error=deprecated-declarations
	-Wno-error=maybe-uninitialized
	-Wno-error=null-dereference
	-Wno-error=old-style-cast
	-Wno-error=shadow
	-Wno-error=sign-conversion
	-Wno-error=undef
	-Wno-error=unknown-pragmas
	-Wnon-virtual-dtor
	-Wnull-dereference
	-Wold-style-cast
	-Woverloaded-virtual
	-Wpedantic
	-Wrange-loop-construct
	-Wshadow
	-Wsign-conversion
	-Wswitch-enum
	-Wundef
	-Wuninitialized
	-Wvarargs
	-Wvexing-parse
	-flax-vector-conversions
	-fopenmp
	-pedantic
	-pedantic-errors)

set(SAMURAI_KOKKOS_GNU_CXX_COMPILE_OPTIONS ${SAMURAI_KOKKOS_COMMON_COMPILE_OPTIONS})

set(SAMURAI_KOKKOS_Clang_CXX_COMPILE_OPTIONS ${SAMURAI_KOKKOS_COMMON_COMPILE_OPTIONS})

########################################################################
####                            Options                             ####
########################################################################

option(SAMURAI_KOKKOS_ENABLE_SANITIZERS "Enable sanitizers in Debug builds"  OFF)
option(SAMURAI_KOKKOS_BUILD_DOC         "Build Doxygen documentation"        OFF)
option(SAMURAI_KOKKOS_USE_NATIVE_OPT    "Enable -march=native -mtune=native" OFF)

########################################################################
####                          Dependencies                          ####
########################################################################

find_package(Kokkos REQUIRED COMPONENTS kokkos)
find_package(KokkosKernels REQUIRED COMPONENTS kokkoskernels)

include(FetchContent)
FetchContent_Declare(samurai GIT_REPOSITORY git@github.com:hpc-maths/samurai.git GIT_TAG main FIND_PACKAGE_ARGS NAMES samurai)

FetchContent_MakeAvailable(samurai)

########################################################################
####                         SubDirectories                         ####
########################################################################

add_subdirectory(benchmarks)

########################################################################
####                            Targets                             ####
########################################################################

add_library(SAMURAI-KOKKOS-Expermients_include INTERFACE)

add_library(SAMURAI-KOKKOS-Expermients_options INTERFACE)

add_library(samurai_kokkos)

########################################################################
####                            Sources                             ####
########################################################################

target_sources(samurai_kokkos PRIVATE
	src/samurai_kokkos_environment.cpp
	src/samurai_kokkos_all_offsets_environment.cpp
	src/csr_matrix.cpp)

target_include_directories(SAMURAI-KOKKOS-Expermients_include INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

########################################################################
####                              Link                              ####
########################################################################

target_link_libraries(samurai_kokkos
	PUBLIC SAMURAI-KOKKOS-Expermients_include
	PUBLIC SAMURAI-KOKKOS-Expermients_options
	PUBLIC samurai
	PUBLIC Kokkos::kokkos
	PUBLIC KokkosKernels::kokkoskernels)

########################################################################
####                        Compile options                         ####
########################################################################

target_compile_features(SAMURAI-KOKKOS-Expermients_options INTERFACE cxx_std_20)

target_compile_options(SAMURAI-KOKKOS-Expermients_options INTERFACE
	$<$<CXX_COMPILER_ID:GNU>:${SAMURAI_KOKKOS_GNU_CXX_COMPILE_OPTIONS}>
	$<$<CXX_COMPILER_ID:Clang>:${SAMURAI_KOKKOS_Clang_CXX_COMPILE_OPTIONS}>)

if(SAMURAI_KOKKOS_ENABLE_SANITIZERS)
	target_compile_options(SAMURAI-KOKKOS-Expermients_options INTERFACE $<$<CONFIG:Debug>:${SAN_FLAGS}>)
endif()

if(SAMURAI_KOKKOS_USE_NATIVE_OPT)
	target_compile_options(SAMURAI-KOKKOS-Expermients_options INTERFACE -march=native -mtune=native)
endif()
