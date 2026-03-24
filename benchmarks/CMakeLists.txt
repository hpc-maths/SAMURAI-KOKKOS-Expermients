########################################################################
####                    benchmarks subdirectory                     ####
########################################################################

########################################################################
####                          Dependencies                          ####
########################################################################

find_package(benchmark REQUIRED)

include(FetchContent)


########################################################################
####                            Targets                             ####
########################################################################

add_library(benchmarks_include INTERFACE)

add_executable(benchmark_axpy)

add_executable(benchmark_axpy2)

add_executable(benchmark_projection)

########################################################################
####                            Sources                             ####
########################################################################

target_sources(benchmark_axpy PRIVATE src/benchmark_axpy.cpp)

target_sources(benchmark_axpy2 PRIVATE src/benchmark_axpy2.cpp)

target_sources(benchmark_projection PRIVATE src/benchmark_projection.cpp)

target_include_directories(benchmarks_include INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

########################################################################
####                      Compile Definitions                       ####
########################################################################

########################################################################
####                              Link                              ####
########################################################################

target_link_libraries(benchmark_axpy
	PRIVATE benchmarks_include
	PRIVATE samurai_kokkos
	PRIVATE benchmark::benchmark)

target_link_libraries(benchmark_axpy2
	PRIVATE benchmarks_include
	PRIVATE samurai_kokkos
	PRIVATE benchmark::benchmark)

target_link_libraries(benchmark_projection
	PRIVATE benchmarks_include
	PRIVATE samurai_kokkos
	PRIVATE benchmark::benchmark)

