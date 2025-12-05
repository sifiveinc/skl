# Conv2D Example: Transforming Convolution into GEMM

This directory demonstrates how to transform 2D convolution operations into General Matrix Multiplication (GEMM) to leverage highly optimized GEMM kernels. The example shows two fundamental approaches for mapping convolution to GEMM operations with automatic selection based on convolution parameters.

## Overview

The example demonstrates two key transformation strategies with intelligent automatic selection:

### 1. **Direct GEMM Mapping** (1x1 Convolution)
For 1x1 convolutions with stride=1 and dilation=1:
- **No preprocessing needed**: Input can be directly reshaped as a matrix
- **Direct matrix multiplication**: `Output = Input × Filter`
- **Optimal performance**: Zero preprocessing overhead
- **Automatic selection**: Chosen when `is_conv2d_to_gemm_im2row_required()` returns `false`

### 2. **Im2Row+GEMM Transformation** (General Convolution)
For general convolutions (any filter size, stride, dilation):
- **Im2Row preprocessing**: Converts input patches into matrix rows
- **GEMM computation**: Transforms convolution into matrix multiplication
- **Automatic selection**: Chosen when `is_conv2d_to_gemm_im2row_required()` returns `true`

## Files

- `main.c`: Test harness demonstrating both transformation approaches with automatic selection
- `conv2d_nhwc_f32.c`: Convolution function implementations
- `conv2d_nhwc_f32.h`: API declarations with detailed documentation
- `conv2d_utils.h`: Utility functions for convolution analysis and automatic transformation selection
- `im2row_hwc.h`: HWC-optimized im2row preprocessing functions
- `README.md`: This documentation

## Convolution to GEMM Transformation

### Data Layout
- **Input/Output**: NHWC format (Batch, Height, Width, Channels)
- **Filter**: HWIO format (Height, Width, Input Channels, Output Channels)

### Transformation 1: Direct GEMM (1x1 Convolution)

When convolution parameters satisfy:
- `filter_height = 1` AND `filter_width = 1`
- `stride_height = 1` AND `stride_width = 1`
- `dilation_height = 1` AND `dilation_width = 1`
- `padding_height = 0` AND `padding_width = 0`

The convolution can be computed as direct matrix multiplication:

```
Input Matrix:  [batch×height×width, input_channels]
Filter Matrix: [input_channels, output_channels]
Output Matrix: [batch×height×width, output_channels]

Output = Input × Filter
```

**Implementation**: `conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390()`
- **No preprocessing**: Input is directly reshaped as matrix
- **Zero overhead**: No memory allocation or data copying
- **Optimal performance**: Direct GEMM computation

### Transformation 2: Im2Row+GEMM (General Convolution)

For general convolutions, the Im2Row transformation converts convolution into GEMM:

```
Im2Row Matrix: [batch × output_height × output_width, filter_height × filter_width × input_channels]
Filter Matrix: [filter_height × filter_width × input_channels, output_channels]
Output Matrix: [batch × output_height × output_width, output_channels]

Output = Im2Row(Input) × Filter
```

**Matrix Dimensions**:
- `M = batch × output_height × output_width`
- `N = output_channels`
- `K = filter_height × filter_width × input_channels`

#### Basic Im2Row Implementation
```c
conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390()
```
- **Im2Row preprocessing**: Extracts input patches into matrix rows
- **GEMM computation**: Uses optimized RISC-V Vector kernel
- **Single-pass**: Processes entire convolution in one GEMM call

### Automatic Transformation Selection

The example includes an intelligent selection mechanism that automatically chooses the optimal transformation approach:

```c
is_conv2d_to_gemm_im2row_required()  // Defined in conv2d_utils.h
```
- **Decision function**: Analyzes convolution parameters to determine optimal transformation
- **Inline implementation**: Efficient compile-time optimization with zero runtime overhead
- **Direct GEMM conditions**: Returns `false` only when ALL conditions are met:
  - `filter_height == 1 && filter_width == 1`
  - `stride_height == 1 && stride_width == 1`
  - `dilation_height == 1 && dilation_width == 1`
  - `padding_height == 0 && padding_width == 0`
- **Im2Row+GEMM conditions**: Returns `true` for all other convolution configurations
- **Runtime selection**: Enables automatic optimization based on convolution characteristics

This decision logic ensures that the most efficient transformation is selected automatically, providing optimal performance for both specialized (1x1) and general convolution cases.

> **Note**: See `conv2d_utils.h` for the complete implementation and detailed documentation of the decision function.

### Reference Implementation
```c
conv2d_io_nhwc_filter_hwio_f32_scalar()
```
- **Scalar reference**: Naive implementation for correctness verification
- **Nested loops**: Direct convolution computation without optimization

### Im2Row Preprocessing

The `im2row_hwc.h` header provides two functions for converting HWC layout convolution patches into matrix rows:

#### `extract_patch_to_row_generic_hwc()`
A straightforward implementation that processes elements individually:
- **Element-by-element processing**: Iterates through each element in the patch
- **HWC layout optimization**: Designed specifically for Height-Width-Channels tensor layout
- **Flexible patch extraction**: Supports arbitrary starting coordinates and patch sizes
- **Bounds checking**: Handles zero-padding for out-of-bounds accesses beyond tensor boundaries
- **Dilation support**: Implements dilated convolution patterns
- **Generic type support**: Works with any primitive data type via `void*` and `element_size`

#### `extract_patch_to_row_hwc()`
An optimized implementation that leverages bulk memory operations:
- **Bulk memory operations**: Uses `memcpy` for efficient channel-wise data movement
- **HWC layout optimization**: Exploits channel-contiguous memory layout in HWC tensors
- **Three-phase processing**: Handles head, middle (full channels), and tail portions separately
- **Memory efficiency**: Minimizes function call overhead through bulk operations
- **Channel optimization**: Optimized for contiguous channel data in HWC layout
- **Generic type support**: Same as generic version, works with any primitive data type via `void*` and `element_size`

> **Note**: See `im2row_hwc.h` for detailed API documentation with complete function signatures, parameter descriptions, and usage examples.

### GEMM Kernel Integration

The example uses the optimized RISC-V Vector GEMM kernel:
- **Function**: `skl_gemm_f32_f32_f32_zve32f_x390`
- **Location**: `src/gemm/rvv/gemm_f32_f32_f32_zve32f_x390.c`
- **Features**:
  - RVV float32 matrix-matrix multiplication (SGEMM) for row-major matrices
  - RISC-V Vector extension (Zve32f) optimized, tuned for X390
  - Supports alpha/beta scaling: `C = alpha × A × B + beta × C`

> **Note**: See `gemm_f32_f32_f32_zve32f_x390.c` for detailed API documentation with complete function signatures, parameter descriptions, and usage examples.

## Performance Considerations

### Why Transform Convolution to GEMM?

**1. Leveraging Highly Optimized Kernels**
- Modern processors have extensively optimized GEMM implementations
- RISC-V Vector GEMM kernels utilize vector instructions for parallel computation
- Single optimized GEMM kernel serves all convolution configurations

**2. Predictable Memory Access Patterns**
- GEMM operations have regular, cache-friendly memory access patterns
- Better cache locality compared to nested convolution loops
- Enables effective prefetching and memory bandwidth utilization

**3. Computational Efficiency**
- Vector instructions process multiple elements simultaneously
- Optimized blocking strategies maximize register utilization
- Reduced instruction overhead compared to nested scalar loops

### Performance Summary

| Transformation | Applicability | Preprocessing | Memory Usage | Performance |
|----------------|---------------|---------------|--------------|-------------|
| **Direct GEMM** | 1x1 filters, stride=1, dilation=1, no padding | None | Minimal | Optimal |
| **Im2Row+GEMM** | Any convolution configuration | Required | Higher | Good |

> **Note**: The higher memory usage of Im2Row+GEMM can be reduced through cache tiling strategies that process the convolution in smaller tiles, trading some performance for controlled memory footprint. This approach enables convolution on large tensors even with limited memory resources.

## Transformation Examples

The main.c example demonstrates both transformation approaches with automatic selection based on convolution parameters:

### Automatic Transformation Selection

The example uses `is_conv2d_to_gemm_im2row_required()` from `conv2d_utils.h` to automatically choose the optimal transformation:

**Direct GEMM Configuration** (when `TEST_IM2ROW_GEMM` is not defined):
- **Input**: `[1, 10, 10, 1024]` (Batch, Height, Width, Channels)
- **Filter**: `[1, 1, 1024, 512]` (Height, Width, Input Channels, Output Channels)
- **Output**: `[1, 10, 10, 512]` (Batch, Height, Width, Channels)
- **Stride**: `[1, 1]` (Height, Width)
- **Padding**: `[0, 0]` (Top, Left padding distances)
- **Dilation**: `[1, 1]` (Height, Width)
- **Matrix Dimensions**: M=100 (10x10), N=512, K=1024 (1×1×1024)
- **Implementation Tested**:
  - Direct GEMM: `conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390`

**Im2Row+GEMM Configuration** (when `TEST_IM2ROW_GEMM` is defined):
- **Input**: `[1, 10, 10, 1024]` (Batch, Height, Width, Channels)
- **Filter**: `[3, 3, 1024, 512]` (Height, Width, Input Channels, Output Channels)
- **Output**: `[1, 5, 5, 512]` (Batch, Height, Width, Channels)
- **Stride**: `[2, 2]` (Height, Width)
- **Padding**: `[0, 0]` (Top, Left padding distances)
- **Dilation**: `[1, 1]` (Height, Width)
- **Matrix Dimensions**: M=25 (5x5), N=512, K=9216 (3×3×1024)
- **Implementation Tested**:
  - Im2Row+GEMM: `conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390`

## Configuration Options

### Build Configuration
The example can be built with different configurations:

**Testing and Benchmarking Options:**
- `ENABLE_TEST`: Includes correctness verification against reference implementation
- `ENABLE_BENCHMARK`: Includes performance measurement using cycle counters

**Transformation Selection Options:**
- **Default configuration** (no `TEST_IM2ROW_GEMM`): Tests Direct GEMM transformation
  - Uses 1x1 filter, stride=1, dilation=1, no padding
  - Automatically selects `conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390`
- **`TEST_IM2ROW_GEMM` defined**: Tests Im2Row+GEMM transformation
  - Uses 3x3 filter, stride=2, dilation=1, no padding
  - Automatically selects `conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390`

Both implementations are tested for numerical accuracy with configurable tolerance levels. The automatic selection mechanism ensures the optimal transformation is chosen based on convolution parameters.
