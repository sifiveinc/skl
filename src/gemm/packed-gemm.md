# SKL Packed GEMM kernels
The general matrix multiplication (GEMM) kernels in SKL sometimes constrain the memory layout of the input matrices by requiring particular transpositions (as described in the [SKL GEMM README](README.md)).
In some cases, the use of specialized instructions or integration into certain software frameworks requires more complex memory layouts, where fixed-size 2-D sub-matrices--blocks--of each matrix are packed into contiguous memory.

This document describes a generic API for such kernels, and how it is specialized for each target.
It also describes the packing and unpacking routines that convert matrices between row-major layout and the packed formats.

## Table of Contents
1. [Packing and Padding](#packing-and-padding)
   - [Example: Matrix A Packing with Padding](#example-matrix-a-packing-with-padding)
   - [Stride Relationships](#stride-relationships)
2. [APIs for Packed GEMM Kernels](#apis-for-packed-gemm-kernels)
   - [Generic Packed GEMM Semantics (Reference Implementation)](#generic-packed-gemm-semantics-reference-implementation)
   - [Naming Convention](#naming-convention)
3. [APIs for Packing & Unpacking Kernels](#apis-for-packing--unpacking-kernels)
4. [Application to Specific ISAs and Frameworks](#application-to-specific-isas-and-frameworks)
   - [Xsfvqdotq: Pack B-Matrix K-Dimension (4xN)](#xsfvqdotq-pack-b-matrix-k-dimension-4xn)
   - [Xsfmm + IREE Framework: Pack M- & N-Dimensions (TExTE)](#xsfmm--iree-framework-pack-m---n-dimensions-texte)
   - [Xsfvqmaccqoq: Packed (M=4)x(K=8)x(N=4) Block-Matrix Products](#xsfvqmaccqoq-packed-m4xk8xn4-block-matrix-products)
5. [Cache Tiling with Packing](#cache-tiling-with-packing)

## Packing and Padding

A "packed" matrix has one or both dimensions partitioned into fixed-size blocks.
For example, if we have a matrix `A` of size `m` x `k`, we might need to partition it into blocks of size `m0` x `k0`, where these dimensions are usually determined by the target hardware instruction.

Since the original matrix dimensions might not be exact multiples of the block size, we must pad the matrix with zeros to make the dimensions match.
The resulting packed matrix `A_pack` will consist of `m1` x `k1` blocks, where `m1 = ceil(m/m0)` and `k1 = ceil(k/k0)`, with each block having size `m0` x `k0` elements.
(While it might be possible to construct packed formats that avoid padding, we exclude this possibility from the API for simplicity, as it is unlikely to be useful in practice.)

### Example: Matrix A Packing with Padding

Consider a 7×6 matrix A with block size `m0=3`, `k0=4` (admittedly an unlikely layout, but useful for illustration):

```
Original Matrix A (7×6):               Conceptual Blocking:
┌─────────────────────────────────┐    ┌───────────┬───────────┐
│ a00 a01 a02 a03 a04 a05  0   0  │    │ Block     │ Block     │
│ a10 a11 a12 a13 a14 a15  0   0  │    │ (0,0)     │ (0,1)     │
│ a20 a21 a22 a23 a24 a25  0   0  │    │ 3×4       │ 3×2+pad   │
│                                 |    ├───────────┼───────────┤
│ a30 a31 a32 a33 a34 a35  0   0  │    │ Block     │ Block     │
│ a40 a41 a42 a43 a44 a45  0   0  │    │ (1,0)     │ (1,1)     │
│ a50 a51 a52 a53 a54 a55  0   0  │    │ 3×4       │ 3×2+pad   │
│                                 |    ├───────────┼───────────┤
│ a60 a61 a62 a63 a64 a65  0   0  │    │ Block     │ Block     │
│  0   0   0   0   0   0   0   0  │    │ (2,0)     │ (2,1)     │
│  0   0   0   0   0   0   0   0  │    │1+pad×4    │1+pad×2+pad│
└─────────────────────────────────┘    └───────────┴───────────┘
        (padded to 9×8)                    m1=3, k1=2 blocks

Packed Layout in Memory (row-major blocks):
Block(0,0): [a00 a01 a02 a03] [a10 a11 a12 a13] [a20 a21 a22 a23]
Block(0,1): [a04 a05  0   0 ] [a14 a15  0   0 ] [a24 a25  0   0 ]
Block(1,0): [a30 a31 a32 a33] [a40 a41 a42 a43] [a50 a51 a52 a53]
Block(1,1): [a34 a35  0   0 ] [a44 a45  0   0 ] [a54 a55  0   0 ]
Block(2,0): [a60 a61 a62 a63] [ 0   0   0   0 ] [ 0   0   0   0 ]
Block(2,1): [a64 a65  0   0 ] [ 0   0   0   0 ] [ 0   0   0   0 ]
```


#### Stride Relationships

For a matrix `A` with packed storage the following stride parameters are defined:
- `rsa0`: row stride within a block: distance in memory between rows in the same block, in units of elements. (Example above: `rsa0 = k0 = 4`)
- `csa0`: column stride within a block: distance in memory between columns in the same block, in units of elements. (Example above: `csa0 = 1`)
- `rsa1`: row stride between blocks: distance in memory between the start of consecutive blocks in the same block column, in units of elements. (Example above: `rsa1 = m0 * k0 * k1 = 24`)
- `csa1`: column stride between blocks: distance in memory between the start of consecutive blocks in the same block row, in units of elements. (Example above: `csa1 = m0 * k0 = 12`)

Generally the naming scheme for these parameters follows the format `{r,c}s{buffer}{layer}`, where `buffer` is the buffer the parameter applies to and `layer` is the blocking layer the parameter describes.

The memory address (in units of elements) for element at block (bi, bj), position (i, j) is:
```
address = base + bi * rsa1 + bj * csa1 + i * rsa0 + j * csa0
```

Thus, the standard row- and column-major layouts are simply special cases of this packed storage, where the block size is 1x1 and inter-block strides are 1.

## APIs for Packed GEMM Kernels

The general form of the packed GEMM API is:
```c
void skl_gemm_packed_<specialization>_<datatypes>_<isa>_<cpu>(
    size_t m0,              // Num. rows in a block of A and C
    size_t n0,              // Num. columns in a block of B and C
    size_t k0,              // Num. columns in a block of A,
                            // rows in a block of B
    size_t m1,              // Num. row blocks in A and C
    size_t n1,              // Num. column blocks in B and C
    size_t k1,              // Num. column blocks in A,
                            // row blocks in B
    <type_c> alpha,         // Scaling factor for A*B
    const <type_a>* a_pack, // Input matrix A
    size_t rsa0,            // Row stride within a block of A
    size_t csa0,            // Column stride within a block of A
    size_t rsa1,            // Row stride between blocks of A
    size_t csa1,            // Column stride between blocks of A
    const <type_b>* b_pack, // Input matrix B
    size_t rsb0,            // Row stride within a block of B
    size_t csb0,            // Column stride within a block of B
    size_t rsb1,            // Row stride between blocks of B
    size_t csb1,            // Column stride between blocks of B
    <type_c> beta,          // Scaling factor for C
    <type_c>* c_pack,       // Output matrix C
    size_t rsc0,            // Row stride within a block of C
    size_t csc0,            // Column stride within a block of C
    size_t rsc1,            // Row stride between blocks of C
    size_t csc1             // Column stride between blocks of C
);
```

The general form assumes all three matrices have already been packed and are stored in packed format in memory.
Usually, all arguments ending in `0` are fixed by the target--and thus omitted from a specific kernel's API--while those ending in `1` are determined by the problem size.
However, in some cases not every dimension will be packed, in which case the original `m`, `n`, or `k` will be passed (see the [Xsfvqdotq](#xsfvqdotq-pack-b-matrix-k-dimension-4xn) example below).

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
    <type_c>* c_block = c_pack + ii1 * rsc1 + jj1 * csc1;
    for (size_t kk1 = 0; kk1 < k1; ++kk1) {
      const <type_a>* ap_block = a_pack + ii1 * rsa1 + kk1 * csa1;
      const <type_b>* bp_block = b_pack + kk1 * rsb1 + jj1 * csb1;
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
- `f32p<...>` for a packed, row-major matrix
- `f32p<...>c` for a packed matrix with blocks in column-major format
- `f32cp<...>c` for a packed matrix with blocks in column-major format and columns of blocks in column-major format
- `f32rcp<...>` for a packed matrix in either block-row-major or block-column-major format, but whose blocks are themselves row-major

When one dimension is not packed, its specifier dimension is set to `1`: `p4x1` for instance.
For block-row-major layouts (no `c` or `rc` before the `p`), if the last dimension is `1`, then it is followed by `c` to emphasize that elements within the block are in column-major order (although this is only vacuously true); when the first dimension is `1`, then its only difference from a non-packed matrix is that it is padded to a multiple of the block length along the packed dimension.
For block-column-major, an inverted but analogous logic applies.

## APIs for Packing & Unpacking Kernels
Packing and unpacking must be performed by the user, and is not automatically applied by the GEMM functions for a variety of reasons:
- The packing may be impossible to perform in-place, requiring a separate buffer that the caller must allocate
- If one input is available offline, or reused across multiple GEMMs, it should not be packed multiple times
- Some applications may provide data in the necessary format already, and should not be forced to repack it

For more details, see the [SKL Matrix Packing Kernels](../pack/README.md) document.

## Application to Specific ISAs and Frameworks

We present three different specializations of the packed GEMM API below, corresponding to different ISAs and frameworks.
In order, they demonstrate packing of 1, 2, and 3 dimensions, respectively.

### Xsfvqdotq: Pack B-Matrix K-Dimension (4xN)

The `sf.vqdot.vx` instruction computes a partial dot product:
```
C[i, j:j+VL] += A[i, k:k+4] * B[k:k+4, j:j+VL]
```
Where the element types of `A` and `B` are `int8_t` and `C` is accumulated as `int32_t`.
Here, `VL` is the vector length, and the value `A[i, k:k+4]` is held in a single 32-bit scalar register.

In order to make use of this instruction with accumulator as a row vector across the output matrix `C`, we must pack the `B` matrix such that each block has dimensions `4xN`, where `N` is the input matrix width.
This is done by dividing the `k` dimension into `k0=4` and `k1=ceil(k/4)`, while keeping the `n` dimension unchanged.
Moreover, since the dot product is performed along the `k` dimension, it must be contiguous in memory, resulting in a column-major layout within each block.

In the actual implementation, we think of blocks as having fixed dimensions `4x1` (`n0=1`) instead of `4xN`.
Each column of the `4xN` blocks of B can be thought of as one of these `4x1` blocks, so `n1 = N`.
And since each `4xN` block of B is column-major and stored contiguously in memory, we find that `rsb0=1` and `csb1=k0*n0`.
Note that `csb0` is unused since each `4x1` block has only one column.

In the A matrix, the "packed" dimension in `p1x4` simply indicates that the row length has been _padded_ to a multiple of 4.
Ideally, if the original `k` was a multiple of 4 or the A matrix was already padded, it can be used without modification.

The packed kernel could be depicted as implemented by the following wrapper for the general packed GEMM API:
```c
void skl_gemm_a1b01_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
  size_t m,             // Num. rows in A and C
  size_t n,             // Num. columns in B and C
  size_t k1,            // Num. 1x4 k blocks in A and 4x1 blocks in B
  const int8_t* a_pack, // Padded  matrix A [m x ceil(k/4) x 4].
  size_t rsa1,           // Row stride between blocks of A
  size_t csa1,           // Column stride between blocks of A
  const int8_t* b_pack, // Packed matrix B [ceil(k/4) x n x (4 x 1)]
  size_t rsb1,          // Row stride between blocks of B
  int32_t* c,           // Output matrix C [m x n]
  size_t rsc,           // Row stride of C
  bool accum            // Whether to accumulate into C
) {
  skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
    1, 1, 4, m, n, k1,      // m0, n0, k0, m1, n1, k1
    1,                      // alpha
    a_pack, 0, 1, rsa1, csa1,   // a_pack, rsa0, csa0, rsa1, csa1
    b_pack, 1, 0, rsb1, 4,  // b_pack, rsb0, csb0, rsb1, csb1
    accum ? 1 : 0,          // beta
    c, 0, 0, rsc, 1         // c_pack, rsc0, csc0, rsc1, csc1
  );
}
```

### Xsfmm + IREE Framework: Pack M- & N-Dimensions (TExTE)

The SiFive matrix engine (Xsfmm) provides a transposed fat outer-product instruction `sf.mm` to compute:
```
C[i:i+TM, j:j+TN] += A[k:k+TK, i:i+TM]^T * B[k:k+TK, j:j+TN]
```
The types of the matrices may vary, but the output matrix is held in matrix register tiles of size `TE` x `TE`, while the input operands are held in vector register groups of size `TK` x `TE`.
The accumulator dimensions `TM` and `TN` are set by the application, and must be <= `TE`, the machine tile size.
The dot product length `TK` is either 1, 2, or 4 depending on the datatype and the application `K` dimension.

__To use the Xsfmm instructions, the `A` matrix must be transposed, but it is not otherwise necessary to pack any matrix into blocks.__
However, integration with the IREE framework imposes additional requirements that motivate a `TExTE` packed layout for this target.

Since the A matrix must be transposed, it is sensible to integrate SKL kernels via the 4D matrix matrix-transpose [MMT4D interface](https://iree.dev/community/blog/2021-10-13-matrix-multiplication-with-mmt4d/) of [the `linalg` dialect](https://iree.dev/community/blog/2021-10-13-matrix-multiplication-with-mmt4d/) (**warning**: the `M` vs `M0` notation in MMT4D is different from the above, and `M` is equivalent to `m1` here).
The 4D layout of MMT4D requires all three matrices be packed into `M0` x `N0`, `M0` x `K0`, or `K0` x `N0` blocks.
However, in the case of the dot product dimension, we simply set the SKL GEMM `k0=1` and `k1` to MMT4D's `K * K0`, meaning that a single call to the GEMM kernel processes a full _panel_ of the A/B matrices, covering the full `K` dimension as if it were a single block.

In theory, it should be possible to avoid exposing this packing to the GEMM kernel itself.
If all matrices were packed into `TE` x `TE` blocks, then the kernel could be called in a loop nest over the packed blocks, applying one `sf.mm` instruction per block.
However, to obtain peak performance it is often necessary for the kernel to use `2*TE` x `2*TE` register tiles, so the strides between blocks must necessarily be exposed to the implementation, motivating the provision of a packed API for this target:

```c
void skl_gemm_a1b01_f32ptex1c_f32cp1xte_f32rcptexte_xsfmm32a32f(
  size_t m1,            // Num. row blocks in A and C
  size_t n1,            // Num. column blocks in B and C
  size_t k,             // Num. columns in A, rows in B
  const float* a_pack,  // Input matrix A [m1 x k x (TE x 1)]
  size_t rsa1,          // Row stride between panels of A
  const float* b_pack,  // Input matrix B [k x n1 x (1 x TE)]
  size_t csb1,          // Column stride between panels of B
  float* c_pack,        // Output matrix C [m1 x n1 x (TE x TE)]
  size_t rsc1,          // Row stride between blocks of C
  size_t csc1,          // Column stride between blocks of C
  bool accum            // Whether to accumulate into C
);
```

The matrix specifiers should be interpreted as:
- A: `f32ptex1c` TE x 1 column vectors of A packed in block-row-major order
- B: `f32cp1xte` 1 x TE row vectors of B packed in block-column-major order
- C: `f32rcptexte` TE x TE blocks of C packed in either block-row- or block-column-major order, depending on `rsc1` and `csc1`

In this case, the packing is performed by the IREE framework itself, and the SKL kernel is called directly with the packed matrices; because of the arbitrary inter-block strides of `C`, it is up to the framework how the individual blocks are arranged with respect to one another.

### Xsfvqmaccqoq: Packed (M=4)x(K=8)x(N=4) Block-Matrix Products

The `sf.vqmacc.4x8x4` instruction computes a fat dot product:
```
C[i:i+4, j:j+4] += A[i:i+4, k:k+8] * B[k:k+8, j:j+4]
```
Where the element types of `A` and `B` are `int8_t` and `C` is accumulated as `int32_t`.
As in the `sf.vqdot.vx` example, `VL` is the vector length, but here the `C` matrix operand is a 4x4 sub-matrix, while `A` and `B` are 4x8 and 8x4 sub-matrices, respectively.

In order to make use of this instruction, all matrices must be packed such that `m0=4`, `n0=4`, and `k0=8`.
Within each block, the `sf.vqmacc.4x8x4` instruction requires a row-major layout, so `rsa0=8`, `rsb0=4`, `rsc0=4`, and all three column strides are `1`.

The specific kernel provided by SKL also assumes a block-row-major layout for the packed matrices, meaning that `rsa1=(k1 * 32)`, `csa1=32`, `rsb1=(n1 * 32)`, `csb1=32`, `rsc1=(n1 * 16)`, and `csc1=16`.

The packed kernel's API is thus:
```c
void skl_gemm_a1b01_i8p4x8_i8p8x4_i32p4x4_xsfvqmaccqoq(
  size_t m1,            // Num. row blocks in A and C
  size_t n1,            // Num. column blocks in B and C
  size_t k1,            // Num. column blocks in A, row blocks in B
  const int8_t* a_pack, // Packed matrix A [m1 x k1 x (4 x 8)]
  size_t rsa1,          // Stride between block-rows of packed A
  const int8_t* b_pack, // Packed matrix B [k1 x n1 x (8 x 4)]
  size_t rsb1,          // Stride between block-rows of packed B
  int32_t* c_pack,      // Packed matrix C [m1 x n1 x (4 x 4)]
  size_t rsc1,          // Stride between block rows of packed C
  bool accum            // Whether to accumulate into C
);
```

The block row strides `rs{a,b,c}1` allow this kernel to perform a partial multiplication on some portion of a larger problem size.

## Cache Tiling with Packing

It may sometimes be desirable to combine explicit cache blocking with packing.
However, it is neither necessary nor desirable to fuse them into a single kernel.
Instead, the combination of calls to packing, GEMM, and unpacking kernels can be tiled in the same way as any other GEMM.
This allows tiling to be tuned to the geometry of the cache hierarchy, rather than being constrained to matrix instruction or register tile sizes, although optimal performance may require them to be multiples to avoid excessive fringe-case handling and padding.

Since SKL's packing functions already copy data, they can be used to implement blocked GEMM by simply calling them on each tile.
For example, the code below shows an example of blocked GEMM with packing using the `xsfvqdot` kernel.
```c
const size_t m_tile = 256; // Application-chosen tile sizes
const size_t n_tile = 128;
const size_t k_tile = 128;

// Need to ensure A-matrix is at least 4-byte aligned,
// so just used 64-byte alignment for both.
#define ALIGN _Alignas(64)

ALIGN static int8_t b_tile[k_tile * n_tile];  // Workspace for packed B
ALIGN static int8_t a_tile[k_tile * m_tile];  // Workspace for padded A

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

        // Pack and pad B tile:
        // Pack into [k1 x nt] x [ 4 x  1] col-major, block-row-major layout
        //        =  [k1 x n1] x [k0 x n0]
        size_t rsb1 = 4 * nt;
        size_t csb1 = 4;
        // Use specialized packing function for Xsfvqdot ISA
        skl_pack_e8_e8rcp4x1c_zve32x(kt, nt, b + kk_tile * rsb + jj_tile, rsb,
                                     b_tile, rsb_1, pad);

        // Pad A tile to ensure k0 * k1 width (multiple of 4):
        // "Pack" into [mt x k1] x [ 1 x  4] block-row-major layout
        //           = [m1 x k1] x [m0 x k0]
        size_t m0 = 1;
        size_t rsa0 = 0; // Unused: only one row in block
        size_t csa0 = 1;
        size_t rsa1 = k0 * k1;
        size_t csa1 = k0;
        // Use generic "packing" function to tile/pad A
        skl_pack_e8_e8rcprc_zve32x(mt, kt, a + ii_tile * rsa + kk_tile, rsa, 1,
                                   m0, k0, a_tile, rsa0, csa0, rsa1, csa1, pad);

        // Multiply A and B tiles, accumulate into un-tiled C
        skl_gemm_a1b01_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
            mt, nt, k1, a_tile, rsa1, csa1, b_tile,
            rsb1, csb1, c + ii_tile * rsc + jj_tile, rsc,
            kk_tile > 0 /* accum */);
      }
    }
  }
}
```
