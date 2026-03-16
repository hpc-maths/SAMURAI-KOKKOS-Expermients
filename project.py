import symgen
import sys

compileOptions = [
	"-fopenmp",
	"-Wundef",
	"-Wvarargs",
	"-Wall",
	"-Wextra",
	"-Winit-self",
	"-Wpedantic",
	"-Werror",
	"-Wconversion",
	"-Wuninitialized",
	"-Wmissing-declarations",
	"-Wsign-conversion",
	"-Wshadow",
	"-Wcast-align",
	"-Wnull-dereference",
	"-Wformat=2",
	"-flax-vector-conversions",
	"-pedantic",
	"-pedantic-errors",
	"-Wno-error=array-bounds",
	"-Wno-c99-extensions",
	"-Wdouble-promotion",
	"-Wswitch-enum",
	"-Wrange-loop-construct",
	"-Wnon-virtual-dtor",
	"-Wold-style-cast",
	"-Woverloaded-virtual",
	"-Wvexing-parse",
	"-Wno-error=array-bounds",
	"-Wno-error=unknown-pragmas",
	"-Wno-error=shadow",
	"-Wno-error=deprecated-declarations",
	"-Wno-error=null-dereference",
	"-Wno-error=maybe-uninitialized",
	"-Wno-error=sign-conversion"]
	
project = symgen.Project("SAMURAI-KOKKOS-Expermients") \
	.set_cmake_prefix("SAMURAI_KOKKOS") \
	.add_language(symgen.Language.CXX) \
	.add_standard(symgen.Language.CXX, 20) \
	\
	.add_compile_options(symgen.Language.CXX, [symgen.Compiler.GNU, symgen.Compiler.CLANG], compileOptions) \
	\
	.add_dependency(symgen.Package("samurai") \
		.set_git("git@github.com:hpc-maths/samurai.git", "main")) \
	.add_dependency(symgen.Package("Kokkos")) \
	\
	.add_executable(symgen.Executable("samurai_with_kokkos") \
		.add_source("src/main.cpp") \
		.add_dependency("samurai") \
		.add_dependency("Kokkos::kokkos"))

project.to_cmake_lists("CMakeLists.txt")
