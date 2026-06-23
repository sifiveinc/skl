# Reference Transpose Implementations

This directory contains reference implementations for matrix transpose operations.
These are scalar implementations used for correctness verification and as reference for optimized ISA-specific implementations.

## Overview

Matrix transpose is a fundamental linear algebra operation that converts an M×N matrix to an N×M matrix by swapping rows and columns.
These reference implementations provide correct, portable behavior across all platforms.

Currently supports 8-bit, 16-bit, and 32-bit element transpose operations.

## Constraints

### Data Layout
- Both A and A^T must be row-major
- Row strides must be specified in elements

### Memory Requirements
- Input and output matrices must not overlap

### Data Type Support
- Works with any data type of the specified element size through type-punning
- 8-bit kernels support any 8-bit data type (int8_t, uint8_t, etc.)
- 16-bit kernels support any 16-bit data type (int16_t, uint16_t, float16_t, bfloat16_t, etc.)
- 32-bit kernels support any 32-bit data type (int32_t, uint32_t, float, etc.)

## Reference Implementations

### **`skl_transpose_e8_ref`**
- Generic scalar implementation for 8-bit elements

### **`skl_transpose_e16_ref`**
- Generic scalar implementation for 16-bit elements

### **`skl_transpose_e32_ref`**
- Generic scalar implementation for 32-bit elements

## Function Signatures

```c
void skl_transpose_e8_ref(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                          size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat);

void skl_transpose_e16_ref(size_t m, size_t n, const uint16_t *SKL_RESTRICT a,
                           size_t rsa, uint16_t *SKL_RESTRICT at, size_t rsat);

void skl_transpose_e32_ref(size_t m, size_t n, const uint32_t *SKL_RESTRICT a,
                           size_t rsa, uint32_t *SKL_RESTRICT at, size_t rsat);
```
