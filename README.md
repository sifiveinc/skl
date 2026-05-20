# SiFive Kernel Library (SKL)

The SiFive Kernel Library (SKL, pronounced "skill") is an open-source collection of high performance routines, referred to as *kernels*, that provide a drop-in replacement for performance-critical components of machine learning (ML) and artifcial intelligence (AI) workloads running on SiFive or other RISC-V-based vector CPUs.

## Features

SKL provides a set of low-level performance primitives that take advantage of the RISC-V vector ISA as well as SiFive-specific extensions to accelerate ML/AI workloads.
Its features include:
- Minimal dependencies: SKL is written in a combination of C with RISC-V vector intrinsics and inline assembly and does not depend on any external libraries.
- Minimal system assumptions: SKL kernels do not make any assumptions about the underlying runtime environment, such as the presence of an operating system or the availability of dynamic memory allocation.
- Decomposability: SKL kernels do not depend on each other and can be used independently; extracting a single kernel requires copying only three files: a kernel-specific header, a source file, and a common header.
- Minimal build system: SKL is intended to be integrated into and even sub-moduled by other projects, so it provides only a minimal CMake-based build system for demonstration purposes, and should easily integrate into other build systems.
- Demonstration code: SKL provides a minimal test program for each kernel that demonstrates its operation and performance.

The functionality of many kernels included in SKL fall into one of the following categories:
- Linear algebra: matrix-matrix and matrix-vector multiplication.
- Nonlinear: exponential; softmax; activation functions.
- Data movement: matrix packing, unpacking, and transposition.
- Convolution: depthwise 2D convolution.

Supported datatypes vary across kernels but may include IEEE 754 double-, single- and half-precision floating point, brain floating point, Open Compute Project 4- and 8-bit floating point, and integer types.

The latest changes and newest features in SKL can be found in our [changelog](./CHANGELOG.md).

## RISC-V Architecture Support & CPU Targets

All SKL kernels depend on the minimal vector extension `zve32x`, though most require additional extensions such as vector floating-point support (`zve64f`) or support for other data types (`zvfh` for half-precision floating point, `zvfbfmin` for brain floating point, etc.).

The source tree is organized by groups of related kernels, such as `gemm` for matrix multiplication and `softmax` for the softmax function.
Separate variants of each kernel class are provided for more specific ISA extensions when available, such as `xsfmm` for SiFive's matrix engine, or `xsfvfexp32e` for SiFive's 32-bit exponential function instruction.
Such ISA requirements are appended to the kernel name, as described in the [API design document](./src/README.md).

Some of the ISA extensions for which SKL currently provides kernels include:
- `Xsfmm{base, 32a8i, 32a16f, 32a32f}`: SiFive's matrix engine.
- `Xsfvfexp{16,32}e`: SiFive's 16-bit and 32-bit exponential function instructions.
- `Xsfvfbfexp16e`: SiFive's 16-bit brain floating point exponential function instruction.
- `Xsfvfexpa`: SiFive's exponential approximation function instruction.
- `Xsfvfbfa`: SiFive's native brain floating point arithmetic instructions.
- `Xsfvqdotq`: SiFive's 8-bit integer 4-element partial dot product instruction.
- `Zvfbfmin`: Minimal support for brain floating point (conversion instructions).
- `Zvfh`: IEEE half-precision floating point arithmetic.
- `Zvfofp8min`: Conversion instructions for 8-bit OFP formats.
- `Zvfofp4min`: Conversion instructions for 4-bit OFP formats.

Additionally, some kernels are tuned separately for specific micro-architectures, such as `x390` for SiFive's X390 processor, in which case the CPU name is also appended to the kernel name.

Users are expected to choose the appropriate kernel variant for their target hardware, based on ISA support and possibly microarchitecture-specific performance tuning.

### Reference Routines

To explain the semantics of each kernel, SKL separately provides a set of reference routines that implement the same functionality as the vectorized kernels, but in pure scalar C code.
These are affixed with the `_ref` suffix, built into `libskl-ref.a`, and provided by the `skl-ref.h` header.
They are both intended as documentation and for use in correctness testing and debugging, but are not intended to provide high performance, even on scalar hardware.
Each reference routine defines a _family_ of kernels that share the same underlying functionality, but may be specialized for particular ISA and CPU targets, or specfic combinations of parameters.
The optimized kernels in a given family are described as calls to the reference routine with particular parameters fixed to specific values.

## Usage

SKL is intended to be used either as a collection of standalone kernels which can be individually exported to other projects, or as a static library.

Compiling SKL kernels requires a C compiler that supports the relevant ISAs for those kernel(s), as well as any corresponding C-language intrinsics.
The Clang toolchain is highly recommended, though the upstream version may not have support for all SiFive-specific extensions used in SKL.

It is strongly recommended to compile SKL functions with `-O3` for maximum performance.
Additionally, appropriate `-march` strings should be used to enable the necessary ISA extensions, as illustrated by the example test programs described in the [testing and benchmarking section](#testing-and-benchmarking) below.

### As a Static Library

SKL's collection of kernels can be compiled into a single static library for use as `-lskl`, and used with the single header `skl.h` that provides the declarations of all SKL kernel functions.

For projects that support CMake version 3.21 or higher, SKL provides a basic build system consisting of a single, top-level [CMakeLists.txt](./CMakeLists.txt), which can be used to produce a static library `libskl.a` as a library dependency.

The CMake build system can also be invoked manually using the following commands from the root SKL directory:
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE="cmake/riscv.cmake"
cmake --build build
```

The toolchain file `cmake/riscv.cmake` is provided for convenience, but it defines a maximal set of RISC-V extensions and may need to be modified for a given target by disabling some items in `SKL_ARCH_EXTENSIONS`:
```cmake
# cmake/riscv.cmake
set(SKL_ARCH_EXTENSIONS
  rv64gcv
  zba
  zbb
  xsfmmbase
  xsfmm32a8i
  xsfmm32a16f
  xsfmm32a32f
  xsfmm64t
  zfh
  zvfh
  xsfvfexpa
  xsfvfexp32e
  zvfofp8min0p2
  zvfofp4min0p1
  xsfvqdotq
)
```
It also presupposes the existence of a Clang toolchain `riscv64-unknown-elf-clang` that is discoverable by CMake.

Whenever significant deviations from the above assumptions are required, users are encouraged to write their own toolchain files or substitute a more appropriate build system.

### As Individual Kernels

SKL's kernels can also be used individually by copying just a few source files into the target project.
To use a particular kernel, copy the following three files from the SKL repository into the target project:
- The kernel's source file (e.g., `src/gemm/rvv/skl_gemm_f32_f32_f32_zve32f_x390.c`)
- The kernel's declaration header (e.g., `src/gemm/rvv/skl_gemm_f32_f32_f32_zve32f_x390.h`)
- The common header `include/skl-common.h`, which must be in the include search path for the kernel's source file.

Indvidual kernel source files should only be added to the build if the target system supports the ISA extensions required by that kernel, and will result in a preprocessor error otherwise.
Similarly, their declaration headers will fail if included in a translation unit that does not support the necessary ISA extensions.

### As a Header-Only Library

SKL kernels can be used as header-only inline functions by defining `SKL_FUNC` and `SKL_FUNC_PRIVATE` as `static inline` before including kernel source files.

```c
#define SKL_FUNC static inline
#define SKL_FUNC_PRIVATE static inline
#include "src/gemm/rvv/skl_gemm_f32_f32_f32_zve32f.c"
```

C++ projects must wrap includes in `extern "C"` blocks:
```cpp
extern "C" {
#define SKL_FUNC static inline
#define SKL_FUNC_PRIVATE static inline
#include "src/gemm/rvv/skl_gemm_f32_f32_f32_zve32f.c"
}
```

**Note**: This usage model is experimental and may require additional integration work.

## Testing and Benchmarking

SKL provides a set of test programs for each family of kernels that can be used to test their operation and performance.
These can be located within the `test/` directory.
Each kernel family (defined by a reference routine) has a corresponding test program that can exercise all of its variants, as supported by a given compilation target.
Test programs are composed of three components:
a global test _driver_, a testing harness specific to a limited set of kernels, and a test suite file defining an array of tests to be ran;
see the [test README](test/README.md) for a more detailed explanation of the testing framework.
Each suite generally contains tests falling into two categories: correctness tests and performance benchmarks.
These categories can be globally enabled via preprocessor definitions, as described below.

Unlike the kernels themselves, the test programs may have multiple dependencies within SKL, and make use of standard library I/O features such as `printf` and `rand`.
Correctness tests are suitable for execution in emulators such as QEMU for demonstration purposes,
but benchmarks should be restricted to execution on actual hardware or cycle-accurate simulators.

### Test Mode (`-DSKL_ENABLE_TESTS=TRUE`)

The test mode of a SKL kernel tester demonstrates proper usage of each variant by constructing appropriate inputs, computing a reference result using the reference routine, and comparing the result of the vector routine to the reference.
It is not intended to be an exhaustive validation suite, but rather a simple illustration of how the kernel can be used.

Test mode is designed for cross compilation on a host machine that is equipped with [QEMU](https://www.qemu.org/) specfically.
SKL's minimal CMake build system can be used to cross-compile the tests using the included toolchain file `cmake/riscv.cmake`, and then execute them in QEMU with the appropriate `-cpu` option to enable the necessary ISA extensions.
(Users of other emulators will need to write their own build and run scripts.)

To build SKL for testing on QEMU, ensure to define the `SKL_ENABLE_TESTS` variable during CMake configuration:
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE="cmake/riscv.cmake" -DSKL_ENABLE_TESTS=TRUE
cmake --build build
cd build/
ctest
```

Test programs are defined via a testing suite, such as test/gemm/rvv/gemm/skl_gemm_f32_f32_f32_zve32f_x390.c, that specify the testing parameters to use during execution.
Tests parameters can be easily reconfigured within such test suites, and new tests can be easily added to the suite by extending the test array appropriately.

### Benchmark Mode (`-DSKL_ENABLE_BENCHMARKS=TRUE`)

Each test suite may also define one or more representative benchmarks with problem parameters depicting an ideal performance regime for each kernel variant.
However, SKL cannot supply the requisite simulation environment for benchmarking, and users are responsible for integrating SKL's benchmarking code into their own simulation or execution environment.

Performance benchmarks can be enabled by defining the `SKL_ENABLE_BENCHMARKS` variable during configuration (which can be defined alongside `SKL_ENABLE_TESTS` if desired):
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE="cmake/riscv.cmake" -DSKL_ENABLE_BENCHMARKS=TRUE
cmake --build build
cd build/
ctest
```

## Documentation

SKL provides two kinds of documentation to assist developers in using SKL.

Many directories contain a `README.md` file that gives a summary of the files and kernels residing in that directory, and possible also in subdirectories.
More comprehensive documents can be found in the `doc` directory.

Each SKL function that is exposed in the SKL API (i.e., those declared with the `SKL_FUNC` attribute) also has a [Doxygen](https://www.doxygen.nl/index.html) documentation string in its associated header which describes the function's operation and its arguments.
These strings can be referred to directly from the source file, or can be compiled into a document using the following Doxygen command (requires Doxygen to be installed) from the `doc` directory:
```shell
cd doc
doxygen Doxyfile
```

## Contributing

Contributions to SKL are welcome! See our [contributing guide](./CONTRIBUTING.md) to get started.

## License

SKL's license can be located [here](./LICENSE.txt).
