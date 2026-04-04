# SKL Matrix Multiplication (GEMM) Kernels

SKL contains a set of general matrix-matrix multiplication (GEMM) kernels for different ISAs and data types, optimized for different target machines.
This document outlines the design principles behind the GEMM collection, and describes the common features of their APIs.
These interfaces conform to the more general specification described in the [SKL Kernels Overview](../README.md) document.

The SKL GEMM source directory contains both public and internal APIs.
The public APIs are intended for end-users, while the internal APIs are for SKL developers and may change without notice.

The rest of this document describes the design principles behind the public GEMM APIs, their naming conventions, source organization, and intended usage patterns.
Some SKL GEMM implementations require packing of matrices into fixed-size blocks.
These APIs are described in the [Packed GEMM API](packed-gemm.md) document, which presupposes familiarity with the concepts and terminology introduced here.

## BLAS-Inspired Generic APIs & Specializations

Currently, all matrix multiplication kernels in SKL provide an interface inspired by the [BLAS standard](https://www.netlib.org/blas), though with some modifications to simplify the description of memory layouts.

All GEMM kernels compute a matrix product `C = alpha * A * B + beta * C`, where `A`, `B`, and `C` are matrices of appropriate dimensions, and `alpha` and `beta` are scalar constants.
Throughout this document, we use capital letters such as 'C' to refer to matrices in the abstract, and lower case letters such as 'c' to refer to specific buffer pointers.
Public APIs are derived from the general form:
```c
void skl_gemm_<specialization>_<datatypes>_<isa>_<cpu>(
    size_t m,               // Number of rows in A and C
    size_t n,               // Number of columns in B and C
    size_t k,               // Number of columns in A, rows in B
    <type_c> alpha,         // Scaling factor for A*B
    const <type_a>* a,      // Input matrix A
    size_t rsa,             // Stride between rows of A
    size_t csa,             // Stride between columns of A
    const <type_b>* b,      // Input matrix B
    size_t rsb,             // Stride between rows of B
    size_t csb,             // Stride between columns of B
    <type_c> beta,          // Scaling factor for C
    <type_c>* c,            // Output matrix C
    size_t rsc,             // Stride between rows of C
    size_t csc              // Stride between columns of C
);
```
Thus, they support arbitrary matrix sizes and memory layouts (row and column strides), as well as alpha- and beta scaling of the matrix product and accumulator.

### Fused Alpha/Beta Scaling
Inspired by the BLAS standard, SKL GEMM APIs support alpha- and beta-scaling of the matrix product and accumulator.
This addition was chosen primarily to support both accumulation (beta=1) and initialization (beta=0), as well as for its general familiarity.
Note that when `k=0`, the `alpha` parameter is ignored, and the `beta` parameter is used to scale the accumulator.

While many other operations could be combined with matrix multiplication, such as requantization and activation functions, SKL does not currently provide APIs for these specialized operations.
For such applications, users can either extract the relevant inner loop nests and extend them, or set `alpha=1` and `beta` to an appropriate value, and tile the accumulator matrix to apply any desired post-processing operations while preserving data locality within the tile.

### Specialization & Constraints
Some targets constrain certain of these parameters to specific values.
For example, matrix engine (Xsfmm) kernels currently require `alpha=1` and `beta=0`, and transpose the A matrix.
Others require specific memory layouts, such as packing or transposition.
These constraints are indicated in the function name, as described below (e.g. `skl_gemm_a1b01_f32c_f32_f32_xsfmm`).

__When a kernel's specialization fixes a parameter to a specific value, that parameter is omitted from the API.__
Since most kernels, aside from scalar reference functions, support only row-major matrices, they do not have both `rsa` and `csa` parameters, for example.

### Naming Convention

Functions follow the general pattern outlined in the [SKL API design document](../README.md).

Regardless of whether the datatype is floating-point or integer, the prefix `skl_gemm` should still be used so long as the overall API roughly matches the general form above, even if some paramters are constrained to specific values.

#### `<datatypes>` Field
The `<datatypes>` field should specify types of all three matrices--even if they are all the same--in function argument order (A, B, C) and separated by underscores.

Each matrix type specifier is further decomposed as `<type>[<packed?>][<col-major?>]`:
- `<type>`: one of the [type specifiers](../README.md#type-specifier-convention) defined in the SKL API design document
- `<packed?>`: `p` if the matrix is packed, omitted otherwise
- `<col-major?>`: `c` if the matrix is stored in column-major format (row stride = 1), `rc` if it may be row-major or column-major (both strides are passed), omitted otherwise
The `rc` specifier is only used in scalar reference functions provided for testing purposes.

#### `<specialization>` Field
The `<specialization>` field is used only to indicate alpha/beta scaling constraints as `[a<value>][b<value>]`.
By default, it is assumed that alpha and beta may be any value, and so this field is omitted:
- ` ` indicates that alpha and beta may be any value
- `a1` indicates that alpha=1, and beta may be any value
- `b0` indicates that beta=0, and alpha may be any value
- `a1b0` indicates that alpha=1 and beta=0
- `a1b01` indicates that alpha=1 and beta=0 or 1
When beta may be either 0 or 1, the `beta` parameter is replaced by a boolean "accumulate"  parameter (`accum`), which is `true` if beta=1 and `false` if beta=0.

Specialization does _not_ currently extend to fixed-size matrix dimensions such as those resulting from register tiling.
Public SKL GEMM functions are dispatch functions that choose between one or more internal kernels, as described below.

#### Examples:
- `skl_gemm_f32rc_f32rc_f32rc_ref(m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)`
- `skl_gemm_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c, rsc)`
- `skl_gemm_i8_i8_i32_zve32x_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c, rsc)`
- `skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f(m, n, k, a, csa, b, rsb, c, rsc, accum)`
- `skl_gemm_a1b01_f32pc_f32p_f32p_xsfmm32a32f(...)`  (See the [Packed GEMM API](packed-gemm.md) document for details)
- `skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq(...)` (See above.)

### Memory Layout and Packing

Some targets require specific memory layouts, such as packing or transposition to make use of specialized instructions.
The details of such APIs are described at much greater length in the [Packed GEMM API](packed-gemm.md) document.

Since most applications will not already have data arranged in the necessary format, SKL provides a set of packing functions for each required layout.
These are separate kernels, and not automatically applied by the GEMM functions for a variety of reasons:
- The packing may be impossible to perform in-place, requiring a separate buffer that the caller must allocate
- If one input is available offline, or reused across multiple GEMMs, it should not be packed multiple times
- Some applications may provide data in the necessary format already, and should not be forced to repack it

As a general rule, the packing functions are named after the target and ISA they are designed for (e.g., `skl_pack_b_i8_xsfvqdotq`).
To the extent possible, their layouts depend only on ISA parameters, not on register tiling decisions or machine-specific parameters such as VLEN or cache sizes.

## File Organization

Due to the large number of targets and data types supported by SKL GEMM kernels, the `skl/src/gemm/` directory is further subdivided by ISA:
```
skl/src/gemm/
├── rvv/
│   ├── ...
└── xsfmm/
    ├── ...
...
```

Note that, because of the number of subsets of the "V" extension, all such kernels are placed in a single `rvv/` directory.

## Implementation Details & Performance Considerations

SKL GEMM kernels do not expose register tiling parameters as part of their public API.
They also do not implement cache blocking or other outer-loop optimizations.
Although both of these facts are internal implementation details, they have important performance implications for users.

### Register Tiling

Most public SKL GEMM functions are effectively dispatch functions that choose between one or more internal register-tiled GEMMs based on problem size and other characteristics.
These individual tilings have been initially excluded from the formal API, as their identities and interfaces are still being refined.

Moreover, integration of GEMM kernels into external frameworks is facilitated by the fact that every public function can complete a full GEMM problem of any size.
However, client applications that tile the output matrix may wish to take the specific internal problem sizes into account to avoid excessive fringe-case handling when one of the dimensions is small.
The only sure way to do so is to inspect the source code inside the dispatch function.

The specific internal tilings supplied by SKL will vary depending on ISA and target machine.
For many ISAs, such as RVV, internal specialized kernels are likely but not guaranteed to include:
- **Big in-cache GEMM**: Default register tiling (e.g., 4xm4) for matrices that are much larger than the machine datapath in all dimensions but small enough to be cache-resident.
- **Small-M GEMM**: Optimized for M=1 cases (otherwise called "GEMV").
- **Small-N GEMM**: Optimized for matrix widths that are not much larger than the datapath width, and cannot benefit from temporal vector execution.

Generally, the names of these internal functions indicate the type of specialization (in addition to any required by the target already) in terms of register tile dimensions:
- `skl_gemm_4x1m4x1_f32_f32_f32_zve32f_x390` for a 4x1 register tile (4 rows, 1 LMUL-4 column)
- `skl_gemm_2x1m2x1_f32_f32_f32_zve32f_x390` for a 2x1 register tile (2 rows, 1 LMUL-4 column)
- `skl_gemm_a1b01_2tm2tn_f32c_f32_f32_xsfmm32a32f` for a 2TEx2TE register tile (`2*TE` rows, `2*TE` columns)

### Cache Blocking and Outer-Loop Nest Optimizations

#### Tiling for Cache Locality
The GEMM kernels in SKL are tuned for cache-resident problems, and do not implement any implicit or explicit cache blocking.
For larger problem sizes, it is expected that users or client frameworks will wrap calls to the SKL kernel in additional outer loops that tile the output matrix--and potentially copy explicit blocks to reduce access strides--with parameters chosen to match the cache geometry.

This is true even in the case of GEMMs that require packed layouts, where the packing function itself can be used to peform explicit cache blocking, as described in the [Packed GEMM API](packed-gemm.md#cache-tiling-with-packing) document.

#### Tiling for Fused Operations
While SKL does not presently offer matrix multiplication kernels that fuse operations beyond alpha- and beta-scaling, it is possible simulate the performance of such fusion by tiling the output matrix appropriately, and interleaving other operations between tiles.

Although this does not permit fused operations to be performed on values that reside in registers, so long as the tile data fits in local memory, much of the performance benefit can be realized.
This means for example that requantization can be performed on a tile of data after it is computed, but before the next tile is loaded.
In such cases, the tile size should be tuned so that the wider intermediate results fit in cache.

#### Tiling in the Dot Product (K) Dimension
In the presence of alpha- and beta-scaling, tiling along the K-dimension (dot product) must be handled carefully.
It is necessary in such cases to treat the first tile along K separately, since it must be scaled by `beta` before adding the product.
Subsequent tiles can be scaled by `alpha` and added to the accumulator without any additional scaling by setting `beta=1`.
