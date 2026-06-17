# Xsfmm Fused Kernel API
In applications, users might want to perform an operation such as alpha/beta scaling, adding a bias, applying an activation function, etc. after computing a matrix product.
The simplest approach is to have distinct GEMM and post-GEMM kernels.
The former compute a GEMM and store the result to memory, while the latter operate on matrices in memory.
An alternate approach for GEMM kernels using SiFive's Xsfmm extension is to leave the matrix product in the tile state and pass it directly to the post-GEMM operator, which avoids storing and reloading the matrix.
This document describes an API for post-GEMM kernels that allows them to be "fused" in this way to the Xsfmm GEMM kernels.
It is assumed that readers are familiar with [SKL's packed GEMM API](../packed-gemm.md) and SiFive's Xsfmm extension.

## The Fused Kernel API
Fused kernels perform an elementwise operation on a single tile in the Xsfmm tile state and a single block of `C`.
Their API is
```
SKL_XSFMM_IN
void kernel(size_t tm, size_t tn, size_t tss, <type> *c, size_t rsc0,
            size_t csc0, size_t rsc1, size_t csc1, size_t row1, size_t col1,
            void *params);
```
where
- `tss` is the tile subset specifier for the first row (column) of the subtile to be operated on
- `tn` is the length of the row (resp. column) `tss` specifies
- `tm` is the number of rows (resp. columns) in the subtile, i.e. the kernel operates on rows (resp. columns) `tss` to `tss + tm - 1`
- `c + row1 * rsc1 + col1 * csc1` points to the block of `C` to be operated on
- `rsc0` and `csc0` are the block's row and column strides
- `params` points to a struct containing any other parameters the kernel needs.

Let `tss[i, j]` denote the `j`th entry of the row or column specified by `tss + i`, and let `c_block = c + row1 * rsc1 + col1 * csc1`.
Then `tss[i, j]` should correspond to `c_block[i * rsc0 + j * csc0]` under the kernel's elementwise operation.
Note that if `tss` has a column pattern, then the operation is effectively applied to the *transpose* of the subtile.

### Examples
We give some examples below of fused kernels for the `float` type.
Pseudocode is given to illustrate the operation each kernel performs.

#### Alpha/beta scaling
To compute a GEMM `C = alpha * A * B + beta * C`, the user can apply an alpha/beta scaling kernel:
```
typedef struct {
  float alpha;
  float beta;
} skl_alpha_beta_scaling_params_f32_f32_t;

SKL_XSFMM_IN
void skl_gemm_alpha_beta_scaling_f32_f32rcptexterc_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  skl_alpha_beta_scaling_params_f32_f32_t *params_cast =
      (skl_alpha_beta_scaling_params_f32_f32_t *)params;
  float alpha = params_cast->alpha;
  float beta = params_cast->beta;

  float *c_block = c + row1 * rsc1 + col1 * csc1;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j)
      c_block[i * rsc0 + j * csc0] =
          alpha * tss[i, j] + beta * c_block[i * rsc0 + j * csc0];
}
```

#### Adding a bias
The following kernel can be used to compute `C = A * B + bias`, where `bias` is a row vector that is added to each row of `A * B`:
```
typedef struct {
  float *bias;
} skl_add_bias_params_f32_f32_t;

SKL_XSFMM_IN
void skl_gemm_add_bias_f32_f32rcptexterc_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  skl_add_bias_params_f32_f32_t *params_cast =
      (skl_add_bias_params_f32_f32_t *)params;
  float *bias = params_cast->bias;

  float *c_block = c + row1 * rsc1 + col1 * csc1;
  float *bias_block = bias + col1 * ETE;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j)
      c_block[i * rsc0 + j * csc0] = tss[i, j] + bias_block[j];
}
```
Note that if `tss` specifies a column, then the bias is added to the *columns* of the subtile.

#### Computing a global maximum
To compute the maximum value of a matrix product, the following kernel can be used tile-by-tile to update the current maximum:
```
typedef struct {
  float *max;
} skl_matrix_max_params_f32_f32_t;

SKL_XSFMM_IN
void skl_gemm_matrix_max_f32_f32rcptexterc_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, void *params) {
  skl_matrix_max_params_f32_f32_t *params_cast =
      (skl_matrix_max_params_f32_f32_t *)params;
  float *max = params_cast->max;

  float *c_block = c + row1 * rsc1 + col1 * csc1;
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j) {
      if (tss[i, j] > *max)
        *max = tss[i, j];
      c_block[i * rsc0 + j * csc0] = tss[i, j];
    }
}
```

### Fused Kernel Application Functions
SKL provides (private) functions that apply a fused kernel to multiple tiles:
```
typedef void (*fused_<type>_<type>_t)(size_t tm, size_t tn, size_t tss,
                                      <type> *c, size_t rsc0, size_t csc0,
                                      size_t rsc1, size_t csc1, size_t row1,
                                      size_t col1, void *params) SKL_XSFMM_IN;

SKL_XSFMM_IN
SKL_FUNC_PRIVATE
void skl_gemm_apply_fused_<type>_<type>rcptexterc_<isa>(
    size_t m, size_t n, size_t tss, size_t rstss, size_t cstss, <type> *c,
    size_t rsc0, size_t csc0, size_t rsc1, size_t csc1, size_t row1,
    size_t col1, fused_<type>_<type>_t kernel, void *params);
```
`tss`, `rstss`, and `cstss` determine a tile layout analogous to the packed layout for matrices.
The tile specifier of `tss` indicates the upper leftmost tile, while `rstss` and `cstss` are the row and column strides for the tile index.
Examples:
```
tss = 0, rstss = 8, cstss = 4
mt0 mt4
mt8 mt12

tss = 3 << 27, rstss = 1, cstss = 3
mt3 mt6 ... mt12
mt4 mt7 ... mt13
mt5 mt8 ... mt14
```
When `tss` has a row pattern, the application functions apply the kernel to the leading `m` x `n` portion of the tile layout.
Below is an illustration of this for the first tile layout above (`tss = 0`, `rstss = 8`, `cstss = 4`) when `m` and `n` are between `ETE` and `2 * ETE`:
```
 ┌─────────────n──────────────┐
┌┌────────────────┬───────────┬────┐┐
││                │           │    ││
││                │           │    ││
││      mt0       │      mt4  │    │ETE
││                │           │    ││
m│                │           │    ││
│├────────────────┼───────────┼────┤┤
││                │           │    ││
││                │           │    ││
││      mt8       │      mt12 │    │ETE
└├────────────────┼───────────┘    ││
 │                │                ││
 └────────────────┴────────────────┘┘
 └──────ETE───────┴──────ETE───────┘
```
The pseudocode below illustrates the application functions' operation:
```
size_t m1 = (m + ETE - 1) / ETE; // ceil(m / ETE)
size_t n1 = (n + ETE - 1) / ETE; // ceil(n / ETE)
size_t m_avl = m;
for (size_t i1 = 0; i1 < m1; ++i1) {
  size_t tm = m_avl > ETE ? ETE : m_avl;
  size_t n_avl = n;
  for (size_t j1 = 0; j1 < n1; ++j1) {
    size_t tn = n_avl > ETE ? ETE : n_avl;
    kernel(tm, tn, tss + i1 * (rstss << 27) + j1 * (cstss << 27), c, rsc0, csc0,
           rsc1, csc1, row1 + i1, col1 + j1, params);
    n_avl -= tn;
  }
  m_avl -= tm;
}
```
If `tss` has a column pattern, the kernel is applied to the transpose of each subtile and stored to the leading `m` x `n` portion of `C`.
Continuing the example above, the tile state would look like:
```
   ┌──────ETE───────┬──────ETE───────┐
  ┌┌────────────────┬────────────────┐┐
  ││                │                ││ 
  ││                │                │n - ETE
ETE│      mt0       │      mt4       ││
  ││                ├────────────────┤┘
  ││                │                │
  ├├──────────┬─────┼───────────┬────┤┐
  ││          │     │           │    ││
  ││          │     │           │    │n - ETE
ETE│      mt8 │     │      mt12 │    ││
  ││          │     ├───────────┘    │┘
  ││          │     │                │
  └└──────────┴─────┴────────────────┘
   └─m - ETE──┘     └──m - ETE──┘
```
It may be easier to visualize by interpreting `rstss` as the column stride and `cstss` as the row stride:
```
 ┌─────────────m──────────────┐
┌┌────────────────┬───────────┬────┐
││                │           │    │
││                │           │    │
││      mt0       │      mt8  │    │
││                │           │    │
n│                │           │    │
│├────────────────┼───────────┼────┤
││                │           │    │
││                │           │    │
││      mt4       │      mt12 │    │
└├────────────────┼───────────┘    │
 │                │                │
 └────────────────┴────────────────┘
```
So, if `tss` has a column pattern, then the application function transposes this region of the tile state, applies the kernel to each subtile, and then stores them to the leading `m` x `n` portion of `C`.

## Inner Loop Functions
The Xsfmm GEMM inner loop functions accumulate a matrix product `A * B` into the current tile state and pass it directly to a fused kernel.
The number of available tiles in the tile state determines the possible tilings of the `C` matrix.
When `TEW` = 32, there are four available tiles (`mt0`, `mt4`, `mt8`, and `mt12`), which can support 1 x 1, 1 x 2, 1 x 3, 1 x 4, 2 x 1, 3 x 1, 4 x 1, and 2 x 2 tilings of `C`.
When `TEW = 8`, there are 16 tiles (`mt0` to `mt15`), which can support tilings of shape `m1` x `n1`, where `m1 * n1 <= 16`.
SKL provides an inner loop function for each `m1` x `n1` tiling with `m1 <= n1`.
If `m1 > n1`, the `n1` x `m1` inner loop function can be used if combined with transposition; more details are given in [Applying a Tiling to `C`](#applying-a-tiling-to-c).

Since the inner loop functions use Xsfmm instructions to compute `A * B`, they assume `A` is packed into `ETE` x 1 column-major blocks and `B` into 1 x `ETE` row-major blocks.
The inner loop functions have the following API:
```
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_m1xn1_<type>rcptex1c_<type>rcp1xte_<type>_<isa>(
    size_t m, size_t n, size_t k, const <type> *a, size_t rsa1, size_t csa1,
    const <type> *b, size_t rsb1, size_t csb1);
```
As with the fused kernel application functions, these functions will compute partial tiles if `m` or `n` is not a multiple of `ETE`.

Since the fused kernels act directly on the tile state, they must be aware of which tiles each inner loop function writes its output to.
The inner loop functions obey the following tile allocation scheme: if `m1 == n1`, tiles are arranged by increasing index in row-major order; otherwise, if `m1 < n1`, tiles are arranged by increasing index in column-major order.
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

2 x 2       2 x 5
mt0 mt1     mt0 mt2 mt4 mt6 mt8
mt2 mt3     mt1 mt3 mt5 mt7 mt9


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

## Tile State Initialization
Before accumulating a matrix product into the tile state, the tile state must be initialized either by zeroing it out or loading a matrix in from memory.

SKL provides the following (private) functions for zeroing out a portion of the tile state:
```
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_<type>_<type>_xsfmmbase(size_t m, size_t n, size_t tss,
                                           size_t rstss, size_t cstss);
```
These functions zero out the leading `m` x `n` portion of the tile layout determined by `tss`, `rstss`, and `cstss`.
Unlike the fused kernel application functions, these ones ignore the pattern and index of `tss` and only uses its tile specifier.

SKL provides the following (private) functions for loading a packed matrix into the tile state:
```
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_load_<type>rcptexterc_<type>_xsfmmbase(
    size_t m, size_t n, const <type> *c, size_t rsc0, size_t csc0, size_t rsc1,
    size_t csc1, size_t tss, size_t rstss, size_t cstss);
```
These functions load the leading `m` x `n` portion of `C` block-by-block into the tile layout determined by `tss`, `rstss`, and `cstss`.
The following pseudocode illustrates their operation:
```
size_t m1 = (m + ETE - 1) / ETE; // ceil(m / ETE)
size_t n1 = (n + ETE - 1) / ETE; // ceil(n / ETE)
size_t m_avl = m;
for (size_t i1 = 0; i1 < m1; ++i1) {
  size_t tm = m_avl > ETE ? ETE : m_avl;
  size_t n_avl = n;
  for (size_t j1 = 0; j1 < n1; ++j1) {
    const <type> *c_block = c + i1 * rsc1 + j1 * csc1;
    size_t tss_tile = tss + i1 * (rstss << 27) + j1 * (cstss << 27);
    size_t tn = n_avl > ETE ? ETE : n_avl;
    for (size_t i0 = 0; i0 < tm; ++i0) {
      for (size_t j0 = 0; j0 < tn; ++j0)
        tss_tile[i0, j0] = c_block[i0 * rsc0 + j0 * csc0];
    }
    n_avl -= tn;
  }
  m_avl -= tm;
}
```
Note that if `tss` has a column pattern, then the transpose of `C` is loaded into the tile layout with `rstss` interpreted as the column stride and `cstss` as the row stride, similar to the column pattern case for the fused kernel application functions.

## Applying a Tiling to `C`
Now that we have described tile state initialization functions, inner loop functions, and fused kernels, we can put them all together to process parts of `C`.
We will first initialize the tile state with zeros or by loading from `C`, then accumulate `A * B` into the tile state, and finally apply a fused kernel to the tile state.
Applying an `m1` x `n1` tiling when `m1 <= n1` is relatively straightforward since transposition is not required.
We can illustrate with a 1 x 2 tiling when `A`, `B`, and `C` are `float` matrices.
The tile allocation is just `mt0 mt4`.
Initialize the tile state by
```
skl_tile_zero_f32_f32_xsfmmbase(m, n, 0, 0, 4);
```
or
```
skl_tile_load_f32rcptexte_f32_xsfmmbase(m, n, c, rsc0, csc0, rsc1, csc1, 0, 0,
                                        4);
```
`rstss` is set to 0 since the tiling has only one row and hence is unused.
Then accumulate `A * B` into the tile state:
```
skl_gemm_inner_loop_1x2_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(m, n, k, a, rsa1,
                                                               csa1, b, rsb1,
                                                               csb1);
```
And finally apply the fused kernel:
```
skl_gemm_apply_fused_f32_f32rcptexterc_xsfmm32a32f(m, n, 0, 0, 4, c, rsc0, csc0,
                                                   rsc1, csc1, row1, col1,
                                                   kernel, params);
```

Tilings with `m1 > n1` are slightly more complicated because they involve transposition.
Suppose we now want to apply a 2 x 1 tiling to `C`.
To use the 1 x 2 inner loop, we need to use transposition to convert the 2 x 1 GEMM tiling `A * B` into the 1 x 2 tiling `B^T * A^T`.
The latter is now an `n` x `m` x `k` GEMM problem.
We still use the same tile allocation `mt0 mt4`.
If we zero out the tile state, the correct call is
```
skl_tile_zero_f32_f32_xsfmmbase(n, m, 0, 0, 4);
```
On the other hand, if we wish initialize the tile state with values from `C`, we actually need to load its transpose `C^T`.
The tile state should look like:
```
 ┌─────────────m──────────────┐
┌┌────────────────┬───────────┬────┐
││                │           │    │
n│                │           │    │
││      mt0       │      mt4  │    │
└├────────────────┼───────────┘    │
 │                │                │
 └────────────────┴────────────────┘
```
Recall that we can load the transpose of `C` by using a `tss` with a column pattern and interpreting `rstss` as the column stride and `cstss` as the row stride.
So, we would call:
```
skl_tile_load_f32rcptexte_f32_xsfmmbase(m, n, c, rsc0, csc0, rsc1, csc1,
                                        1 << 24 /* mt0 with column pattern */,
                                        4, 0);
```
The inner loop function is called with `A` and `B` swapped and transposed:
```
skl_gemm_inner_loop_1x2_f32rcptex1c_f32rc1xtep_f32_xsfmm32a32f(n, m, k, b, csb1,
                                                               rsb1, a, csa1,
                                                               rsa1);
```
Finally, we apply the fused kernel with a column pattern:
```
skl_gemm_apply_fused_f32_f32rcptexterc_xsfmm32a32f(m, n, 1 << 24, 4, 0, c, rsc0,
                                                   csc0, rsc1, csc1, row1, col1,
                                                   kernel, params);
```
