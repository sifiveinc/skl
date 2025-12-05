# SKL Examples

This directory contains example implementations that demonstrate how to use SKL functions to build higher-level operations and algorithms.

## Purpose

Examples serve as:
- **Educational resources** showing practical usage patterns
- **Reference implementations** for common algorithms
- **Integration guides** demonstrating how to combine multiple SKL functions
- **Performance benchmarks** for evaluating optimization strategies

## Current Examples

### [conv2d/](conv2d/)
Demonstrates efficient 2D convolution implementation using optimized GEMM kernels:
- **Direct GEMM mapping**: Optimal approach for 1x1 convolutions with no preprocessing overhead
- **Im2Row+GEMM transformation**: Shows how to convert general convolutions into matrix multiplication
- **RISC-V Vector optimization**: Uses `skl_gemm_f32_f32_f32_zve32f_x390` kernel tuned for X390
- **Generic type support**: Foundation for multiple data types (currently float32)

> **See [conv2d/README.md](conv2d/README.md) for detailed algorithm explanation, implementation details, and performance analysis.**

## Building Examples

Examples are built alongside tests when `SKL_BUILD_TESTS` or `SKL_BUILD_BENCHMARKS` is enabled:

```bash
cmake -B build -DSKL_BUILD_TESTS=ON -DSKL_BUILD_BENCHMARKS=ON -DCMAKE_TOOLCHAIN_FILE=cmake/riscv.cmake ..
cd build && cmake --build . --verbose
```

## Running Examples

Examples can be executed through CTest:

```bash
# Through CTest (all examples)
ctest -R example
```

## Adding New Examples

To add a new example:

1. **Create directory**: `mkdir example/your_example`
2. **Add source files**: Implement your example with documentation
3. **Update CMakeLists.txt**: Add `skl_add_example()` call
4. **Add README**: Document the example's purpose and usage

### Example CMakeLists.txt Entry
```cmake
skl_add_example(your_example_name your_example/main.c "")
```

## Guidelines

Examples should:
- **Focus on algorithms**: Show how to combine SKL functions for higher-level operations
- **Include documentation**: Provide clear README with algorithm explanation
- **Support testing**: Include correctness verification when `ENABLE_TEST` is defined
- **Support benchmarking**: Include performance measurement when `ENABLE_BENCHMARK` is defined
- **Follow naming**: Use descriptive names that indicate the algorithm and key optimizations
