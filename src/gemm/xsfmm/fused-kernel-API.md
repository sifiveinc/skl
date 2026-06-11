# Xsfmm fused kernel API
In applications, users might want to perform an operation such as alpha/beta scaling, adding a bias, applying an activation function, etc. after computing a matrix product.
This document describes an API for kernels that can be fused to a GEMM operation, i.e. applied directly to the output of a GEMM operation in the tile state.
Assumes reader is familiar with packed GEMM API.

## ABI attributes

## Inner loop functions
The number of available tiles in the Xsfmm tile state determines the possible tilings of the `C` matrix.
When `TEW` = 32, there are four available tiles (`mt0`, `mt4`, `mt8`, and `mt12`), which can support 1 x 1, 1 x 2, 1 x 3, 1 x 4, 2 x 1, 3 x 1, 4 x 1, and 2 x 2 tilings of `C`.
When `TEW = 8`, there are 16 tiles (`mt0` to `mt15`), which can support tilings of shape `m1` x `n1`, where `m1 * n1 <= 16`.
SKL provides an inner loop function for each `m1` x `n1` tilings with `m1 <= n1`.
These inner loop functions accumulate a matrix product `A * B` into the current tile state.
If `m1 > n1`, the `n1` x `m1` inner loop function can be used if combined with transposition.
More details are given in later sections.

Since the inner loop functions use Xsfmm instructions to compute `A * B`, they assume `A` is packed into `ETE` x 1 column-major blocks and `B` into 1 x `ETE` row-major blocks.
The inner loop functions have the following API:
```
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void inner_loop_m1xn1_f32rcpc_f32rcp_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1);
```
The product `A * B` remains in the tile state so that the fused kernel can operate on it directly.
This API handles partial tiles if `m` or `n` is not a multiple of `ETE`.
The inner loop functions obey the following tile allocation scheme: if `m1 == n1`, tiles are arranged by index in increasing order in row-major order; otherwise, if `m1 < n1`, tiles are arranged by index in increasing order in column-major order.
Below are some examples of the tile allocation scheme:
`TEW` = 32
```
1 x 1     1 x 4
mt0       mt0 mt4 mt8 mt12

2 x 2
mt0  mt4
mt8 mt12
```
`TEW` = 8
```
1 x 16
mt0 mt1 mt2 ... mt14 m15

3 x 3           3 x 4
mt0 mt1 mt2     mt0 mt3 mt6  mt9
mt3 mt4 mt5     mt1 mt4 mt7 mt10
mt6 mt7 mt8     mt2 mt5 mt8 mt11

4 x 4
 mt0  mt1  mt2  mt3
 mt4  mt5  mt6  mt7
 mt8  mt9 mt10 mt11
mt12 mt13 mt14 mt15
```

## Fused Kernel API
Fused kernels perform an operation on a tile in the tile state and a block of `C`.
Their API is
```
SKL_XSFMM_IN
void fused(size_t tm, size_t tn, size_t tss, float *c, size_t rsc0,
           size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params);
```
where
- `tss` is a tile subset specifier
- tn is the length of the row (column) `tss` specifies
- `tm` is the number of rows (resp. columns) to be operated on, i.e. rows (resp. columns) `tss` to `tss + tm - 1`
- `c + row1 + rsc1 + col1 * csc1` points to the block of `C`
- `rsc0` is the block's row stride
- `params` points to a struct containing any other parameters the kernel needs.

### Examples
In the examples below, `tss[i, j]` denotes the `j`th entry of the row or column specified by `tss + i`.
Pseudocode is given to illustrate the operation each kernel performs.
#### Alpha/beta scaling
To compute a GEMM `C = alpha * A * B + beta * C`, the user can apply an alpha/beta scaling kernel to `A * B`.
```
typedef struct {
  float alpha;
  float beta;
} alpha_beta_scaling_params_f32_f32_t;

void skl_gemm_alpha_beta_scaling_f32_f32rcp_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  alpha_beta_scaling_params_f32_f32_t *params_cast =
      (alpha_beta_scaling_params_f32_f32_t *)params;
  
  float *c_block = c + row1 * rsc1 + col1 * csc1;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j)
      c_block[i * rsc0 + j] = beta * c_block[i * rsc0 + j] + alpha * tss[i, j];
}
```
Note that if `tss` specifies a column, then the transpose of the tile is scaled and stored to `C`.

#### Adding a bias
The following kernel can be used to compute `C = A * B + bias`, where `bias` is a row vector that is added to each row of `A * B`:
```
typedef struct {
  float *bias;
  size_t csbias1;
} add_bias_params_f32_f32_t;

void skl_gemm_add_bias_f32_f32rcp_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  add_bias_params_f32_f32_t *params_cast = (add_bias_params_f32_f32_t *)params;
  
  float *c_block = c + row1 * rsc1 + col1 * csc1;
  float *bias_block = bias + col1 * csbias1;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j)
      c_block[i * rsc0 + j] = tss[i, j] + bias_block[j];
}
```
Note that if `tss` specifies a column, then the bias is added to each row of the transpose of the tile and the result is stored to `C.
In other words, the bias is added to the *columns* of the tile.

#### Computing a global maximum
To compute the maximum value of the matrix product, the following kernel can be used tile-by-tile to update the current maximum:
```
typedef struct {
  float *max;
} matrix_max_params_f32_f32_t;

void skl_gemm_matrix_max_f32_f32rcp_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  matrix_max_params_f32_f32_t *params_cast =
      (matrix_max_params_f32_f32_t *)params;
  
  float *c_block = c + row1 * rsc1 + col1 * csc1;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j) {
      if (tss[i, j] > *max)
        *max = tss[i, j];
      c_block[i * rsc0 + j] = tss[i, j];
    }
}
```

## Tile State Initialization
Before accumulating any matrix products into the tile state, the tile state must be initialized either by zeroing it out or loading a matrix in from memory.

SKL provides the following (private) function for zeroing out a portion of the tile state:
```
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_f32_f32_xsfmmbase(size_t m, size_t n, size_t tss,
                                     size_t rstss, size_t cstss);
```
`tss`, `rstss`, and `cstss` determine the tile layout.
The tile specifier of `tss` indicates the upper leftmost tile.
`rstss` and `cstss` are the row and column strides for the tile index.
Note that the pattern and index of `tss` are ignored.
This function zeroes out the leading `m` x `n` portion of this tile layout.
Examples:
```
tss = 0, rstss = 8, cstss = 4
mt0 mt4
mt8 mt12
```

```
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_load_f32rcp_f32_xsfmmbase(size_t m, size_t n, const float *c,
                                        size_t rsc0, size_t rsc1, size_t csc1,
                                        size_t tss, size_t rstss,
                                        size_t cstss)
```
This function loads the leading `m` x `n` portion of `C` block-by-block into the tile layout determined by `tss`, `rstss`, and `cstss`.
More specifically, row `i0` of the block at `c + i1 * rsc1 + j1 * csc1` is loaded into the row or column specified by `tss + i1 * rstss + j1 * cstss + i0`.
Note that if `tss` has a column pattern, then the `C` block is transposed when it is loaded into the tile state.
Examples:
```
Original Matrix (7 x 6):            
┌─────────────────────────────────┐
│                │                │ 
│                │                │    
│                │                │
│                │                │
│                │                │ 
│─────────────────────────────────│
│                │                │
│                │                │
│                │                │
│                │                │
│                │                │
└─────────────────────────────────┘
```




The general matrix multiplication (GEMM) kernels in SKL sometimes constrain the memory layout of the input matrices by requiring particular transpositions (as described in the [SKL GEMM README](README.md)).
In some cases, the use of specialized instructions or integration into certain software frameworks requires more complex memory layouts, where fixed-size 2D sub-matrices--blocks--of each matrix are packed into contiguous memory.

This document describes a generic API for such kernels, and how it is specialized for each target.
It also describes the packing and unpacking routines that convert matrices between row-major layout and the packed formats.

## Table of Contents
1. [Packing and Padding](#packing-and-padding)
   - [Example: Matrix Packing with Padding](#example-matrix-packing-with-padding)
   - [Stride Parameters](#stride-parameters)
2. [APIs for Packed GEMM Kernels](#apis-for-packed-gemm-kernels)
   - [Generic Packed GEMM Semantics (Reference Implementation)](#generic-packed-gemm-semantics-reference-implementation)
   - [Naming Convention](#naming-convention)
3. [APIs for Packing & Unpacking Kernels](#apis-for-packing--unpacking-kernels)
4. [Application to Specific ISAs and Frameworks](#application-to-specific-isas-and-frameworks)
   - [Xsfvqdotq: Pack B-Matrix K-Dimension (4 x N)](#xsfvqdotq-pack-b-matrix-k-dimension-4-x-n)
   - [Xsfmm + IREE Framework: Pack M- & N-Dimensions (TE x TE)](#xsfmm--iree-framework-pack-m---n-dimensions-te-x-te)
   - [Xsfvqmaccqoq: Packed (M = 4)x(K = 8)x(N = 4) Block-Matrix Products](#xsfvqmaccqoq-packed-m--4xk--8xn--4-block-matrix-products)
5. [Cache Tiling with Packing](#cache-tiling-with-packing)

## Packing and Padding

A _packed_ matrix has one or both dimensions partitioned into fixed-size blocks.
For example, if we have a matrix of size `m` x `n`, we might need to partition it into blocks of size `m0` x `n0`, where these dimensions are usually determined by the target hardware instruction.

Since the original matrix dimensions might not be exact multiples of the block size, we must pad the matrix so that they are.
Typically, matrices are padded with zeroes, but other values are possible.
The resulting packed matrix will consist of `m1` x `n1` blocks, where `m1 = ceil(m/m0)` and `n1 = ceil(n/n0)`, with each block having size `m0` x `n0` elements.
While it might be possible to construct packed formats that avoid padding, we exclude this possibility from the API for simplicity, as it is unlikely to be useful in practice.

### Example: Matrix Packing with Padding

Consider a 7 x 6 matrix with block size `m0 = 3`, `n0 = 4` (admittedly an unlikely layout, but useful for illustration):

```
Original Matrix (7 x 6):               Conceptual Blocking:
┌─────────────────────────────────┐    ┌───────────┬───────────┐
│ c00 c01 c02 c03 c04 c05  0   0  │    │ Block     │ Block     │
│ c10 c11 c12 c13 c14 c15  0   0  │    │ (0,0)     │ (0,1)     │
│ c20 c21 c22 c23 c24 c25  0   0  │    │ 3×4       │ 3×2+pad   │
│                                 |    ├───────────┼───────────┤
│ c30 c31 c32 c33 c34 c35  0   0  │    │ Block     │ Block     │
│ c40 c41 c42 c43 c44 c45  0   0  │    │ (1,0)     │ (1,1)     │
│ c50 c51 c52 c53 c54 c55  0   0  │    │ 3×4       │ 3×2+pad   │
│                                 |    ├───────────┼───────────┤
│ c60 c61 c62 c63 c64 c65  0   0  │    │ Block     │ Block     │
│  0   0   0   0   0   0   0   0  │    │ (2,0)     │ (2,1)     │
│  0   0   0   0   0   0   0   0  │    │1+pad×4    │1+pad×2+pad│
└─────────────────────────────────┘    └───────────┴───────────┘
        (padded to 9 x 8)                m1 = 3, n1 = 2 blocks

Packed Layout in Memory (row-major blocks):
Block(0,0): [c00 c01 c02 c03] [c10 c11 c12 c13] [c20 c21 c22 c23]
Block(0,1): [c04 c05  0   0 ] [c14 c15  0   0 ] [c24 c25  0   0 ]
Block(1,0): [c30 c31 c32 c33] [c40 c41 c42 c43] [c50 c51 c52 c53]
Block(1,1): [c34 c35  0   0 ] [c44 c45  0   0 ] [c54 c55  0   0 ]
Block(2,0): [c60 c61 c62 c63] [ 0   0   0   0 ] [ 0   0   0   0 ]
Block(2,1): [c64 c65  0   0 ] [ 0   0   0   0 ] [ 0   0   0   0 ]
```

#### Stride Parameters

For a matrix `C` with packed storage, the following stride parameters are defined:
- `rsc0`: row stride within a block, i.e. distance in memory from element `(i, j)` to `(i + 1, j)` in the same block, in units of elements. Example above: `rsc0 = n0 = 4`.
- `csc0`: column stride within a block, i.e. distance in memory from element `(i, j)` to `(i, j + 1)` in the same block, in units of elements. Example above: `csc0 = 1`.
- `rsc1`: row stride between blocks, i.e. distance in memory from the start of block `(bi, bj)` to the start of block `(bi + 1, bj)`, in units of elements. Example above: `rsc1 = m0 * n0 * n1 = 24`.
- `csc1`: column stride between blocks, i.e. distance in memory from the start of block `(bi, bj)` to the start of block `(bi, bj + 1)`, in units of elements. Example above: `csc1 = m0 * n0 = 12`.

Generally the naming scheme for these parameters follows the format `{r,c}s{buffer}{layer}`, where `buffer` is the buffer the parameter applies to and `layer` is the blocking layer the parameter describes.

The memory address (in units of elements) for the element at position `(i, j)` in block `(bi, bj)` is:
```
address = base + bi * rsc1 + bj * csc1 + i * rsc0 + j * csc0
```

Thus, the standard row- and column-major layouts are simply special cases of this packed storage, where the block size is 1 x 1 and `csc1` or `rsc1` is 1, respectively.

## APIs for Packed GEMM Kernels

The general form of the packed GEMM API is:
```c
void skl_gemm_<specialization>_<datatypes>_<isa>_<cpu>(
    size_t m0,         // Num. rows in a block of A and C
    size_t n0,         // Num. columns in a block of B and C
    size_t k0,         // Num. columns in a block of A, rows in a block of B
    size_t m1,         // Num. block-rows in A and C
    size_t n1,         // Num. block-columns in B and C
    size_t k1,         // Num. block-columns in A, block-rows in B
    <type_c> alpha,    // Scaling factor for A * B
    const <type_a>* a, // Packed input matrix A
    size_t rsa0,       // Row stride within a block of A
    size_t csa0,       // Column stride within a block of A
    size_t rsa1,       // Row stride between blocks of A
    size_t csa1,       // Column stride between blocks of A
    const <type_b>* b, // Packed input matrix B
    size_t rsb0,       // Row stride within a block of B
    size_t csb0,       // Column stride within a block of B
    size_t rsb1,       // Row stride between blocks of B
    size_t csb1,       // Column stride between blocks of B
    <type_c> beta,     // Scaling factor for C
    <type_c>* c,       // Packed output matrix C
    size_t rsc0,       // Row stride within a block of C
    size_t csc0,       // Column stride within a block of C
    size_t rsc1,       // Row stride between blocks of C
    size_t csc1        // Column stride between blocks of C
);
```

The general form assumes all three matrices have already been packed and are stored in packed format in memory.
Usually, all arguments ending in `0` are fixed by the target--and thus omitted from a specific kernel's API--while those ending in `1` are determined by the problem size.
However, in some cases not every dimension will be packed, in which case the original `m`, `n`, or `k` will be passed (see the [Xsfvqdotq](#xsfvqdotq-pack-b-matrix-k-dimension-4-x-n) example below).

As in the case of basic GEMM, the various stride parameters suffice to describe arbitrary transpositions (row- and column-major) both of blocks and within them.

### Generic Packed GEMM Semantics (Reference Implementation)
For the sake of completeness, the above API's semantics are defined by the following reference implementation:
```c
// Loops over blocks (done by software)
for (size_t ii1 = 0; ii1 < m1; ++ii1) {
  for (size_t jj1 = 0; jj1 < n1; ++jj1) {
    <type_c> acc[m0 * n0]; // (Usually a register in practice)
    for (size_t idx = 0; idx < m0 * n0; ++idx)
      acc[idx] = 0;
    for (size_t kk1 = 0; kk1 < k1; ++kk1) {
      const <type_a>* ap_block = a + ii1 * rsa1 + kk1 * csa1;
      const <type_b>* bp_block = b + kk1 * rsb1 + jj1 * csb1;
      // Compute block product (loops usually map to 1 instruction)
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          for (size_t kk0 = 0; kk0 < k0; ++kk0) {
            <type_a> a_val = ap_block[ii0 * rsa0 + kk0 * csa0];
            <type_b> b_val = bp_block[kk0 * rsb0 + jj0 * csb0];
            acc[ii0 * n0 + jj0] += a_val * b_val;
          }
        }
      }
    }
    // Update C matrix block (sometimes by specialized stores)
    <type_c>* c_block = c + ii1 * rsc1 + jj1 * csc1;
    for (size_t ii0 = 0; ii0 < m0; ++ii0) {
      for (size_t jj0 = 0; jj0 < n0; ++jj0) {
        size_t c_idx = ii0 * rsc0 + jj0 * csc0;
        c_block[c_idx] = beta * c_block[c_idx] + alpha * acc[ii0 * n0 + jj0];
      }
    }
  }
}
```
Essentially, packing adds additional loop nests to the basic GEMM algorithm.
However, in practice, these innermost loops (`ii0`, `jj0`, and `kk0`) correspond to operations performed by the hardware, and so are effectively unrolled away.

### Naming Convention
The naming convention for packed GEMM APIs is similar to that of the basic GEMM APIs, but uses the `<datatypes>` field to indicate packing.
Specifically, a `p<m0xn0>` specifier is inserted after each matrix type to indicate that it is packed in blocks of size `m0` by `n0` (or `k0` and `n0` etc.).

On either side of `p`, transposition specifiers are used as usual:
- `f32p<...>` for a packed matrix in block-row-major format whose blocks are row-major
- `f32p<...>c` for a packed matrix in block-row-major format whose blocks are column-major
- `f32cp<...>c` for a packed matrix in block-column-major format whose blocks are column-major
- `f32rcp<...>` for a packed matrix in either block-row-major or block-column-major format, but whose blocks are themselves row-major

When one dimension is not packed, the corresponding block dimension is set to 1: `p4x1` for instance.

## APIs for Packing & Unpacking Kernels
Packing and unpacking must be performed by the user, and is not automatically applied by the GEMM functions for a variety of reasons:
- The packing may be impossible to perform in-place, requiring a separate buffer that the caller must allocate
- If one input is available offline, or reused across multiple GEMMs, it should not be packed multiple times
- Some applications may provide data in the necessary format already, and should not be forced to repack it

For more details, see the [SKL Matrix Packing Kernels](../pack/README.md) document.

## Application to Specific ISAs and Frameworks

We present three different specializations of the packed GEMM API below, corresponding to different ISAs and frameworks.
In order, they demonstrate packing of 1, 2, and 3 dimensions, respectively.

### Xsfvqdotq: Pack B-Matrix K-Dimension (4 x N)

The `sf.vqdot.vx` instruction computes a partial dot product:
```
C[i, j:j+VL] += A[i, k:k+4] * B[k:k+4, j:j+VL]
```
where `A` and `B` are `int8_t` matrices and `C` is an `int32_t` matrix.
`VL` is the vector length, `A[i, k:k+4]` is held in a single 32-bit scalar register, and `B[k:k+4, j:j+VL]` is held contiguously in a vector register group in column-major order.

To use this instruction to compute a matrix product, it is natural to pack the `A` matrix into 1 x 4 blocks; pack the `B` matrix into 4 x `N` column-major panels, each stored in contiguous memory; and leave `C` in row-major order.
Thus, only the `K` dimension is packed; it is divided into `k0 = 4` and `k1 = ceil(K/4)`.
Since the `N` dimension is not packed, we set `n0 = 1`, and similarly, we set `m0 = 1`.

Due to its block shape, the simplest way to "pack" the `A` matrix is to just pad the rows to a multiple of 4.
As a special case, if `K` is already a multiple of 4, then `A` can be used without modification.

The packed kernel could be depicted as implemented by the following wrapper for the general packed GEMM API:
```c
void skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
  size_t m,        // Num. rows in A and C
  size_t n,        // Num. columns in B and C
  size_t k1,       // Num. block-columns in A, block-rows in B
  int32_t alpha,   // Scaling factor for A * B
  const int8_t* a, // Packed input matrix A [m x k1 x (1 x 4)].
  size_t rsa1,     // Row stride between blocks of A
  size_t csa1,     // Column stride between blocks of A
  const int8_t* b, // Packed input matrix B [k1 x n x (4 x 1)]
  size_t rsb1,     // Row stride between blocks of B
  int32_t beta,    // Scaling factor for C
  int32_t* c,      // Output matrix C [m x n]
  size_t rsc,      // Row stride of C
) {
  skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
    1, 1, 4, m, n, k1,   // m0, n0, k0, m1, n1, k1
    alpha,               // alpha
    a, 0, 1, rsa1, csa1, // a, rsa0, csa0, rsa1, csa1
    b, 1, 0, rsb1, 4,    // b, rsb0, csb0, rsb1, csb1
    beta,                // beta
    c, 0, 0, rsc, 1      // c, rsc0, csc0, rsc1, csc1
  );
}
```

### Xsfmm + IREE Framework: Pack M- & N-Dimensions (TE x TE)

The SiFive matrix engine (Xsfmm) provides a transposed fat outer-product instruction `sf.mm` to compute:
```
C[i:i+TM, j:j+TN] += A[k:k+TK, i:i+TM]^T * B[k:k+TK, j:j+TN]
```
The types of the matrices may vary, but the output matrix is held in matrix register tiles of size `TE` x `TE`, while each of the `TK` rows of each input operand is held in vector register groups of size `TE`.
The accumulator dimensions `TM` and `TN` are set by the application, and must be <= `TE`, the machine tile size.
The dot product length `TK` is either 1, 2, or 4 depending on the datatype and the application `K` dimension.

__To use the Xsfmm instructions, the `A` matrix must be transposed, but it is not otherwise necessary to pack any matrix into blocks.__
However, integration with the IREE framework imposes additional requirements that motivate a `TE` x `TE` packed layout for this target.

Since the `A` matrix must be transposed, it is sensible to integrate SKL kernels via the 4D matrix matrix-transpose [MMT4D interface](https://iree.dev/community/blog/2021-10-13-matrix-multiplication-with-mmt4d/) of [the `linalg` dialect](https://iree.dev/community/blog/2021-10-13-matrix-multiplication-with-mmt4d/) (**warning**: the `M` vs `M0` notation in MMT4D is different from the above, and `M` is equivalent to `m1` here).
The 4D layout of MMT4D requires all three matrices be packed into `M0` x `N0`, `M0` x `K0`, or `K0` x `N0` blocks.
However, in the case of the dot product dimension, we simply set the SKL GEMM `k0 = 1` and `k1` to MMT4D's `K * K0`, meaning that a single call to the GEMM kernel processes a full _panel_ of the `A`/`B` matrices, covering the full `K` dimension as if it were a single block.

In theory, it should be possible to avoid exposing this packing to the GEMM kernel itself.
If all matrices were packed into `TE` x `TE` blocks, then the kernel could be called in a loop nest over the packed blocks, applying one `sf.mm` instruction per block.
However, to obtain peak performance it is often necessary for the kernel to use `2*TE` x `2*TE` register tiles, so the strides between blocks must necessarily be exposed to the implementation, motivating the provision of a packed API for this target:

```c
void skl_gemm_a1b01_f32ptex1c_f32cp1xte_f32rcptexte_xsfmm32a32f(
  size_t m1,       // Num. block-rows in A and C
  size_t n1,       // Num. block-columns in B and C
  size_t k,        // Num. columns in A, rows in B
  const float* a,  // Packed input matrix A [m1 x k x (TE x 1)]
  size_t rsa1,     // Row stride between panels of A
  const float* b,  // Packed input matrix B [k x n1 x (1 x TE)]
  size_t csb1,     // Column stride between panels of B
  float* c,        // Packed output matrix C [m1 x n1 x (TE x TE)]
  size_t rsc1,     // Row stride between blocks of C
  size_t csc1,     // Column stride between blocks of C
  bool accum       // Whether to accumulate into C
);
```

The matrix specifiers should be interpreted as:
- `A`: `f32ptex1c` `TE` x 1 column vectors of `A` packed in block-row-major order
- `B`: `f32cp1xte` 1 x `TE` row vectors of `B` packed in block-column-major order
- `C`: `f32rcptexte` `TE` x `TE` blocks of `C` packed in either block-row- or block-column-major order, depending on `rsc1` and `csc1`

In this case, the packing is performed by the IREE framework itself, and the SKL kernel is called directly with the packed matrices; because of the arbitrary inter-block strides of `C`, it is up to the framework how the individual blocks are arranged with respect to one another.

### Xsfvqmaccqoq: Packed (M = 4)x(K = 8)x(N = 4) Block-Matrix Products

The `sf.vqmacc.4x8x4` instruction computes a fat dot product:
```
C[i:i+4, j:j+4] += A[i:i+4, k:k+8] * B[k:k+8, j:j+4]
```
where `A` and `B` are `int8_t` matrices and `C` is an `int32_t` matrix.
Each of the three operands is held contiguously in row-major order in its own vector register group.
Packed GEMM kernels using this instruction would have block dimensions `m0 = 4`, `n0 = 4`, and `k0 = 8`; intra-block row strides `rsa0 = 8`, `rsb0 = 4`, `rsc0 = 4`; and all intra-block column strides equal to 1.

For a fixed `A`, `sf.vqmacc.4x8x4` can compute multiple such fat dot products at once provided that the `B` blocks are stored together contiguously in a vector register group and likewise for the `C` blocks.
Because of this, it is natural to use a block-row-major layout for `B` and `C`.
The specific kernel provided by SKL assumes this layout for `A` as well, so `csa1 = 32`, `csb1 = 32`, and `csc1 = 16`.

The packed kernel's API is thus:
```c
void skl_gemm_a1b01_i8p4x8_i8p8x4_i32p4x4_xsfvqmaccqoq(
  size_t m1,       // Num. block-rows in A and C
  size_t n1,       // Num. block-columns in B and C
  size_t k1,       // Num. block-columns in A, block-rows in B
  const int8_t* a, // Packed input matrix A [m1 x k1 x (4 x 8)]
  size_t rsa1,     // Row stride between blocks of A
  const int8_t* b, // Packed input matrix B [k1 x n1 x (8 x 4)]
  size_t rsb1,     // Row stride between blocks of B
  int32_t* c,      // Packed output matrix C [m1 x n1 x (4 x 4)]
  size_t rsc1,     // Row stride between blocks of C
  bool accum       // Whether to accumulate into C
);
```

The block row strides `rs{a,b,c}1` allow this kernel to perform a partial multiplication on some portion of a larger problem size.

## Cache Tiling with Packing

It may sometimes be desirable to combine explicit cache blocking with packing.
However, it is neither necessary nor desirable to fuse them into a single kernel.
Instead, the combination of calls to packing, GEMM, and unpacking kernels can be tiled in the same way as any other GEMM.
This allows tiling to be tuned to the geometry of the cache hierarchy, rather than being constrained to matrix instruction or register tile sizes, although optimal performance may require them to be multiples to avoid excessive fringe-case handling and padding.

Since SKL's packing functions already copy data, they can be used to implement blocked GEMM by simply calling them on each tile.
The code below shows an example of blocked GEMM with packing using the `Xsfvqdotq` kernel.
```c
const size_t m_tile = 256; // Application-chosen tile sizes
const size_t n_tile = 128;
const size_t k_tile = 128;

// To avoid misaligned loads from A, need to ensure A matrix is at least 4-byte
// aligned, so just used 64-byte alignment for both.
#define ALIGN _Alignas(64)

ALIGN static int8_t a_tile[m_tile * k_tile];  // Workspace for padded A
ALIGN static int8_t b_tile[k_tile * n_tile];  // Workspace for packed B

#define min(a, b) ((a) < (b) ? (a) : (b))

/**
 * User-supplied wrapper function for blocked GEMM with packing.
 * Computes C = A * B, where A is m x k, B is k x n, and C is m x n.
 */
void example_blocked_gemm(size_t m, size_t n, size_t k, const int8_t *a,
                          size_t rsa, const int8_t *b, size_t rsb, int32_t *c,
                          size_t rsc) {
  for (size_t ii_tile = 0; ii_tile < m; ii_tile += m_tile) {
    size_t mt = min(m_tile, m - ii_tile);
    for (size_t jj_tile = 0; jj_tile < n; jj_tile += n_tile) {
      size_t nt = min(n_tile, n - jj_tile);
      for (size_t kk_tile = 0; kk_tile < k; kk_tile += k_tile) {
        size_t kt = min(k_tile, k - kk_tile);

        size_t k0 = 4;
        size_t k1 = (kt + 3) / 4;
        uint8_t pad = 0; // Padding value

        // Pad A tile to ensure k0 * k1 row length (multiple of 4):
        // "Pack" into [mt x k1] x [ 1 x  4] block-row-major layout
        //           = [m1 x k1] x [m0 x k0]
        size_t m0 = 1;
        size_t rsa0 = 0; // Unused: only one row in block
        size_t csa0 = 1;
        size_t rsa1 = k0 * k1;
        size_t csa1 = k0;
        // Use generic packing function to pad A
        skl_pack_e8_e8rcprc_zve32x(mt, kt, a + ii_tile * rsa + kk_tile, rsa, 1,
                                   m0, k0, a_tile, rsa0, csa0, rsa1, csa1, pad);

        // Pack and pad B tile:
        // Pack into [k1 x nt] x [ 4 x  1] block-row-major layout
        //        =  [k1 x n1] x [k0 x n0]
        size_t rsb1 = 4 * nt;
        size_t csb1 = 4;
        // Use specialized packing function for Xsfvqdotq ISA
        skl_pack_e8_e8p4x1c_zve32x(kt, nt, b + kk_tile * rsb + jj_tile, rsb,
                                   b_tile, rsb1, pad);

        // Multiply A and B tiles, accumulate into un-tiled C
        skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
            mt, nt, k1, 1 /* alpha */, a_tile, rsa1, csa1, b_tile, rsb1,
            kk_tile ? 1 : 0 /* beta */, c + ii_tile * rsc + jj_tile, rsc);
      }
    }
  }
}
```
