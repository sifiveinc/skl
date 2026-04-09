# Transpose Kernels

This directory contains optimized kernels for matrix transpose operations. Matrix transpose is a fundamental linear algebra operation that converts an M×N matrix to an N×M matrix by swapping rows and columns. These kernels provide efficient implementations for various data types and target architectures.

Currently supports 8-bit, 16-bit, and 32-bit element transpose operations.

## Constraints

### Data Layout
- Both A and A^T must be row-major
- Row strides must be specified in elements

### Memory Requirements
- Input and output matrices must not overlap
- All pointers should be properly aligned for optimal performance on the target architecture

### Data Type Support
- Works with any data type of the specified element size through type-punning
- 8-bit kernels support any 8-bit data type
- 16-bit kernels support any 16-bit data type  
- 32-bit kernels support any 32-bit data type

## Kernel List

### Scalar Implementations
#### **`skl_transpose_e8_ref`**
- Generic implementation for 8-bit elements

#### **`skl_transpose_e16_ref`**
- Generic implementation for 16-bit elements

#### **`skl_transpose_e32_ref`**
- Generic implementation for 32-bit elements

```c
void skl_transpose_e8_ref(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                             size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e16_ref(size_t m, size_t n,
                              const uint16_t *SKL_RESTRICT a, size_t rsa,
                              uint16_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e32_ref(size_t m, size_t n,
                              const uint32_t *SKL_RESTRICT a, size_t rsa,
                              uint32_t *SKL_RESTRICT at, size_t rsat);
```

### RVV Implementations

#### **`skl_transpose_e8_zve32x`**
- Optimized RISC-V Vector (RVV) implementation for 8-bit elements

#### **`skl_transpose_e16_zve32x`**
- Optimized RISC-V Vector (RVV) implementation for 16-bit elements

#### **`skl_transpose_e32_zve32x`**
- Optimized RISC-V Vector (RVV) implementation for 32-bit elements

```c
void skl_transpose_e8_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                             size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e16_zve32x(size_t m, size_t n,
                              const uint16_t *SKL_RESTRICT a, size_t rsa,
                              uint16_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e32_zve32x(size_t m, size_t n,
                              const uint32_t *SKL_RESTRICT a, size_t rsa,
                              uint32_t *SKL_RESTRICT at, size_t rsat);
```

### Xsfmm Implementations

#### **`skl_transpose_e8_xsfmmbase`**
- High-performance implementation using the Xsfmm matrix engine for 8-bit elements

#### **`skl_transpose_e16_xsfmmbase`**
- High-performance implementation using the Xsfmm matrix engine for 16-bit elements

#### **`skl_transpose_e32_xsfmmbase`**
- High-performance implementation using the Xsfmm matrix engine for 32-bit elements

```c
void skl_transpose_e8_xsfmmbase(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e16_xsfmmbase(size_t m, size_t n,
                                 const uint16_t *SKL_RESTRICT a, size_t rsa,
                                 uint16_t *SKL_RESTRICT at, size_t rsat);
void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                 const uint32_t *SKL_RESTRICT a, size_t rsa,
                                 uint32_t *SKL_RESTRICT at, size_t rsat);
```
