# SKL Matrix Packing Kernels

This family of kernels implements the pack and unpack operations for matrices used in the [packed GEMM kernels](../gemm/packed-gemm.md).
It includes a set of packing and, when relevant, unpacking functions for each GEMM-related ISA extension.

These routines reorder the elements of a 2D, row-major matrix of size `m` x `n` into a blocked layout where blocks of size `m0` x `n0` are stored contiguously, and padded with zeros if necessary, for a total of `m1` x `n1` blocks, where `m1 = ceil(m/m0)` and `n1 = ceil(n/n0)`.
Depending on the requirements of the target ISA or application, the elements within each block may also be transposed, and the blocks themselves may be stored in row- or column-major order.
When a matrix is packed along only one dimension and no transposition is implied, then the "packing" operation consists entirely of _padding_ the matrix to a multiple of the block size.

Not all extensions require packing for all matrices, but kernels for such cases may nevertheless be provided, as some applications may require similar layouts across all matrices, or seek locality benefits from arranging data in the shape of the underlying multiplication operation, or one of its dimensions.
The documentation for each kernel indicates whether it is required or optional.
See the [list of packing kernels](#list-of-packing-kernels) for details.

## General API for Packing & Unpacking Kernels

These functions specialize the general form:
```c
void skl_pack_<src type>_<dst_type>_<isa>(
  size_t m,           // Num. rows in input matrix
  size_t n,           // Num. columns in input matrix
  const <type>* src,  // Input matrix
  size_t rs,          // Row stride of input matrix
  size_t cs,          // Column stride of input matrix
  size_t m0,          // Num. rows in a block of the input matrix
  size_t n0,          // Num. columns in a block of the input matrix
  <type>* dst,        // Output packed matrix [m1 x n1]
  size_t rs0,         // Row stride within a block of the output matrix
  size_t cs0,         // Column stride within a block of the output matrix
  size_t rs1,         // Row stride between blocks of the output matrix
  size_t cs1          // Column stride between blocks of the output matrix
) {
  size_t m1 = (m + m0 - 1) / m0; // Num. row blocks in the input matrix
  size_t n1 = (n + n0 - 1) / n0; // Num. column blocks in the input matrix
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const <type>* src_block = src + ii1 * m0 * rs + jj1 * n0 * cs;
      <type>* dst_block = dst + ii1 * rs1 + jj1 * cs1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
            dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0 * cs];
          } else {
            // Pad with zeros
            dst_block[ii0 * rs0 + jj0 * cs0] = 0;
          }
        }
      }
    }
  }
}
```
Unpacking functions work analogously.


As in the case of GEMM kernels, most packing kernels will fully specialize or constrain most of the parameters, and they are thus omitted from the API.
Usually, `m0`, `n0`, `rs0`, and `cs0` are all fixed by the target's layout, and indicated in the kernel's name as part of the datatype specifier of the output matrix.

### Naming Convention for Packing Kernels

The name of the packing kernel indicates the block size and ISA extension required by the kernel, and whether it is required or optional.
The general form is `skl_pack_<src type>_<dst type>_<isa>`.

The datatype specifiers follow the same conventions as in the corresponding [GEMM kernels](../gemm/packed-gemm.md) in which the matrix will be used, except that only the element width is indicated, as the internal format is irrelevant to the packing implementation (e.g. `e32` is used rather than `f32`).

Thus, the `<dst type>` term may further be broken down as `e<sew>[<outer-block-order>]p<m0>x<n0>[<inner-block-order>]`, where `m0` and `n0` are the block dimensions (rows and columns per block, respectively).

It is assumed by default that the ordering of elements within a block is fixed to be row-major.
If instead the layout within a block is column-major, the inner-block-order term is `c`, and if it may be either, `rc` is used.
Similarly, the ordering of blocks relative to each other is assumed to be block-row-major, but if it is either block-column-major or configurable, then either `c` or `rc` is inserted, respectively, before the packing specifier `p`.

When one of the dimensions is `1`, there is technically no difference in memory ordering between row- and column-major; however, the order is still said to be "column-major" when the second dimension is `1`, as this corresponds to the canonical layout of a column vector, and better communicates the transposition implied by the layout.
Use of `1` for either block dimension effectively creates a block with one side of arbitrary size, referred to as a _panel_.

> **Note**: As in other kernel families, the ISA suffix indicates the minimum requirement to execute the packing kernel itself (normally `zve32x`), but its primary intended use may be for a particular matrix extension. This is indicated in the documentation for each kernel (packing and GEMM).
>
> The `skl_pack_e8_e8ptex1c_xsfmmbase` function is an exception, as it uses the matrix engine to accelerate transposition within blocks.

## Packing Layout Examples

Several examples of packed layouts are shown below, with element types omitted.

### Example: p4x1c Block Layout (Block-Row-Major with Inner Transposition)
```
Original row-major matrix (8x4):        Packed layout [m0=4, n0=1, m1=2, n1=4]:

 0  1  2  3                             Block(0,0)  Block(0,1)  Block(0,2)  Block(0,3)
 4  5  6  7                             ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
 8  9 10 11                             │  0  │     │  1  │     │  2  │     │  3  │
12 13 14 15                             │  4  │     │  5  │     │  6  │     │  7  │
16 17 18 19                             │  8  │     │  9  │     │ 10  │     │ 11  │
20 21 22 23                             │ 12  │     │ 13  │     │ 14  │     │ 15  │
24 25 26 27                             └─────┘     └─────┘     └─────┘     └─────┘
28 29 30 31
                                        Block(1,0)  Block(1,1)  Block(1,2)  Block(1,3)
                                        ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
                                        │ 16  │     │ 17  │     │ 18  │     │ 19  │
                                        │ 20  │     │ 21  │     │ 22  │     │ 23  │
                                        │ 24  │     │ 25  │     │ 26  │     │ 27  │
                                        │ 28  │     │ 29  │     │ 30  │     │ 31  │
                                        └─────┘     └─────┘     └─────┘     └─────┘

Memory layout:
[ 0, 4, 8,12,  1, 5, 9,13,  2, 6,10,14,  3, 7,11,15, 16,20,24,28, 17,21,25,29, ...]
 └Block(0,0)┘ └Block(0,1)┘ └Block(0,2)┘ └Block(0,3)┘ └Block(1,0)┘ └Block(1,1)┘

Strides: rs0=1 (row stride within block), cs0=4 (column stride within block)
         rs1=16 (row stride between blocks), cs1=4 (column stride between blocks)
```

### Example: cp1x4 Block Layout (Block-Column-Major with Outer Transposition)
```
Original row-major matrix (4x8):        Packed layout [m0=1, n0=4, m1=4, n1=2]:

 0  1  2  3  4  5  6  7                 Block(0,0)    Block(0,1)
 8  9 10 11 12 13 14 15                 ┌───────────┐ ┌───────────┐
16 17 18 19 20 21 22 23                 │ 0  1  2  3│ │ 4  5  6  7│
24 25 26 27 28 29 30 31                 └───────────┘ └───────────┘

                                        Block(1,0)    Block(1,1)
                                        ┌───────────┐ ┌───────────┐
                                        │ 8  9 10 11│ │12 13 14 15│
                                        └───────────┘ └───────────┘

                                        Block(2,0)    Block(2,1)
                                        ┌───────────┐ ┌───────────┐
                                        │16 17 18 19│ │20 21 22 23│
                                        └───────────┘ └───────────┘

                                        Block(3,0)    Block(3,1)
                                        ┌───────────┐ ┌───────────┐
                                        │24 25 26 27│ │28 29 30 31│
                                        └───────────┘ └───────────┘

Memory layout:
[ 0, 1, 2, 3,  8, 9,10,11, 16,17,18,19, 24,25,26,27,  4, 5, 6, 7, 12,13,14,15, ...]
 └Block(0,0)┘ └Block(1,0)┘ └Block(2,0)┘ └Block(3,0)┘ └Block(0,1)┘ └Block(1,1)┘

Strides: rs0=4 (row stride within block), cs0=1 (column stride within block)
         rs1=4 (row stride between blocks), cs1=16 (column stride between blocks)
```

### Example: p2x2c Block Layout (Block-Row-Major with Column-Major Within Blocks)
```
Original row-major matrix (4x4):        Packed layout [m0=2, n0=2, m1=2, n1=2]:

 0  1  2  3                             Block(0,0)  Block(0,1)
 4  5  6  7                             ┌───────┐   ┌───────┐
 8  9 10 11                             │ 0  4  │   │ 2  6  │
12 13 14 15                             │ 1  5  │   │ 3  7  │
                                        └───────┘   └───────┘

                                        Block(1,0)  Block(1,1)
                                        ┌───────┐   ┌───────┐
                                        │ 8 12  │   │10 14  │
                                        │ 9 13  │   │11 15  │
                                        └───────┘   └───────┘

Memory layout:
[ 0, 4, 1, 5,  2, 6, 3, 7,  8,12, 9,13, 10,14,11,15 ]
 └Block(0,0)┘ └Block(0,1)┘ └Block(1,0)┘ └Block(1,1)┘

Strides: rs0=1 (row stride within block), cs0=2 (column stride within block)
         rs1=8 (row stride between blocks), cs1=4 (column stride between blocks)
```
Note: This layout does not occur in any of the current SKL GEMM kernels.
It is for illustrative purposes only.



## List of Packing Kernels

### Required Packing Kernels
All of these kernels are required for correct operation of the corresponding GEMM kernels.

#### Xsfvqdotq
- `skl_pack_e8_e8p4x1c_zve32x`: Pack the B matrix into 4x1 panels (4x1 block-row-major example above) to use `sf.vqdot.vx` without reduction summation.

#### Xsfvqmaccqoq
These kernels are required for use of the `sf.vqmacc.4x8x4` instruction.
Because of the small block size, significant specialization is required to achieve reasonable performance.
- `skl_pack_e32_e32p4x4_zve32x`: Used for the C matrix.
- `skl_unpack_e32p4x4_e32_zve32x`: Inverse of `skl_pack_e32_e32p4x4_zve32x`.
- `skl_pack_e8_e8p4x8_zve32x`: Used for the A matrix.
- `skl_pack_e8_e8p8x4_zve32x`: Used for the B matrix.

### Optional Packing Kernels
None of these are strictly required for correct operation of the corresponding GEMM kernels, but they may be useful for performance or compatibility reasons.
(For Xsfmm, it is never required to pack any matrix; only [transposition](../transpose/README.md) is required if the A matrix is in row-major format, or B is in column-major format.)
- `skl_pack_e8_e8ptex1c_xsfmmbase`: Transpose (TE x K) panels using the matrix engine. Used on a row-major A matrix or column-major B matrix.

### Packing Kernels That Do Not Exist
These names refer to kernels that users might expect to exist, but do not, because they can be implemented with an appropriate block size in the generic RVV packing kernel at reasonable performance.
In general, so long as the block size is large enough in at least one dimension (as is the case in paneling), a generic packing kernel with a large enough vector length can achieve reasonable performance.

- ~~`skl_pack_e8_e8rcp1xte_zve32x`~~: Pack (K x TE) panels. Used on a column-major A matrix or row-major B matrix.
- ~~`skl_pack_e32_e32rcptexte_zve32x`~~: Pack (TE x TE) blocks without transposition. Used on a C matrix.
- ~~`skl_pack_e8_e8cp1xvlm8_zve32x`~~: Pack (1 x VLMAX*8) column panels.

In addition to being unnecessary for correct execution, these can all be implemented by supplying the machine-dependent dimension (TE or VLMAX) to the generic packing kernel.

## Generic RVV Packing Kernel [**Future Work**]

The fully-general packing function described [above](#general-api-for-packing--unpacking-kernels) can be vectorized with RVV, and will perform reasonably well in many cases.
Specifically, if the input is row-major, the innermost loop can be vectorized, and if `cs0` is 1, unit-stride loads and stores can be used.
So long as the block width `n0 * SEW` is at least as large as the machine data path width, no specialized optimizations should be required.
**This implementation is not yet provided by SKL.**

