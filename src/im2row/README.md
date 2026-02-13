# Im2Row Kernels

The `im2row` directory contains functions for converting HWC layout convolution patches into matrix rows, enabling efficient convolution-to-GEMM transformation.

Im2row preprocessing is only required when any stride > 1, any filter dimension > 1, or any dilation > 1. For 1x1 filters with stride=1 and dilation=1, direct GEMM is more efficient.

**Input tensor layout**: HWC (Height, Width, Channels) - channels are contiguous in memory.

## Im2Row Transformation Overview

Im2row transforms a convolution operation into a matrix multiplication (GEMM):

```
Convolution:  Output[B,H,W,OC] = Conv(Input[B,IH,IW,IC], Filter[FH,FW,IC,OC])
                                    ↓ Im2Row transformation
GEMM:         Output[M,N] = Im2RowMatrix[M,K] × Filter[K,N]
              where M = B×H×W, K = FH×FW×IC, N = OC
```

### Visual Example: 3×3 Filter on 5×5 Input (HWC Layout)

For a single output position, im2row extracts a patch and linearizes it into a row:

```
Input Tensor (H=5, W=5, C=3):          Filter (FH=3, FW=3, IC=3, OC=2):
┌─────────────────────┐                ┌─────────────────────┐
│ [r,g,b][r,g,b]...   │                │ 3×3×3 = 27 weights  │
│ [r,g,b][r,g,b]...   │                │ per output channel  │
│ [r,g,b][r,g,b]...   │                └─────────────────────┘
│ ...                 │
└─────────────────────┘

Im2Row extracts 3×3 patch:             Linearized to row:
┌─────────────────────┐                ┌─────────────────────────────────┐
│[r,g,b][r,g,b][r,g,b]│  ──────────>   │[r,g,b,r,g,b,r,g,b,r,g,b,...]    │
│[r,g,b][r,g,b][r,g,b]│                │ 1 row × 27 elements (K=FH×FW×IC)│
│[r,g,b][r,g,b][r,g,b]│                └─────────────────────────────────┘
└─────────────────────┘

For all output positions:
Im2Row Matrix [M×K]:                   Filter Matrix [K×N]:
┌─────────────────────┐                ┌──────────┐
│ row for out[0,0]    │                │ weights  │
│ row for out[0,1]    │      ×         │ for OC=2 │  = Output [M×N]
│ row for out[0,2]    │                │ channels │
│ ...                 │                └──────────┘
└─────────────────────┘
  M rows × K cols                        K × N
```

### Padding Behavior

When extracting patches near tensor boundaries, im2row handles out-of-bounds accesses by padding with a configurable byte value (typically 0). This implements zero-padding commonly used in convolutions:

```
Example: 3×3 patch at top-left corner with padding=1

Logical padded view:              Actual input tensor:
┌───────────────────────┐         ┌─────────────────┐
│ [0,0,0][0,0,0][0,0,0] │ ← pad   │ [r,g,b][r,g,b]  │
│ [0,0,0][r,g,b][r,g,b] │         │ [r,g,b][r,g,b]  │
│ [0,0,0][r,g,b][r,g,b] │         │ ...             │
└───────────────────────┘         └─────────────────┘
   ↑
  pad

Im2row output for this patch:
┌─────────────────────────────────────────────────────────────────┐
│ [0,0,0][0,0,0][0,0,0][0,0,0][r,g,b][r,g,b][0,0,0][r,g,b][r,g,b] │
│  pad    pad    pad    pad    real   real   pad    real   real   │
└─────────────────────────────────────────────────────────────────┘
  27 elements total (3×3 positions × 3 channels)

Implementation:
- For each position (h,w) in the patch:
  - If (h,w) is within input bounds → memcpy actual data
  - If (h,w) is outside input bounds → memset with zero_byte
```

**Key points:**
- The `zero_byte` parameter controls the padding value (usually 0x00 for zero-padding)
- Padding is applied per-element during patch extraction, not as a preprocessing step
- This approach avoids allocating extra memory for a padded input tensor
- Bounds checking: `(in_h >= 0) && (in_h < input_height) && (in_w >= 0) && (in_w < input_width)`

## Optimization Strategy

The optimized RVV implementations exploit the HWC memory layout where channels are contiguous. Instead of copying elements one-by-one, they use a **three-phase approach** (head-middle-tail) to maximize bulk memory operations:

```
Patch extraction with partial channel start (patch_begin_coord[2] = 5):

Input spatial position (H,W):  [c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11]
                                              ↑                      ↑
                                           start=5                end=12

Output row:  [HEAD: c5..c11] [MIDDLE: full channels] [TAIL: partial channels]
             └─ partial ─┘    └─── bulk memcpy ───┘  └─ partial ─┘

HEAD:   Copy remaining channels from first spatial position (if start offset > 0)
MIDDLE: Bulk copy full channel groups from subsequent spatial positions
TAIL:   Copy partial channels from last spatial position (if needed)
```

**Benefits:**
- **Scalar (`generic_hwc`)**: Copies 1 element at a time → many small memcpy calls
- **Optimized (`hwc_zve32x`)**: Copies full channel groups → fewer, larger memcpy calls
- **Type-safe (`e8/e16/e32`)**: Same optimization, with compile-time type checking

## Kernel List

### Scalar Implementations

#### **`skl_im2row_generic_hwc`**
Generic patch extraction with element-by-element processing for any primitive data type.

```c
void skl_im2row_generic_hwc(
    void *out, const void *in_batch, size_t element_size, int32_t in_w_origin,
    int32_t in_h_origin, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t dilation_width_factor, size_t dilation_height_factor,
    unsigned char zero_byte, const size_t patch_begin_coord[3],
    size_t patch_elements);
```

### RVV Implementations

#### **`skl_im2row_hwc_zve32x`**
Optimized patch extraction with bulk memory operations for any primitive data type.

```c
void skl_im2row_hwc_zve32x(void *out, const void *in_batch, size_t element_size,
                           int32_t in_w_origin, int32_t in_h_origin,
                           size_t input_height, size_t input_width,
                           size_t input_channel, size_t filter_height,
                           size_t filter_width, size_t dilation_width_factor,
                           size_t dilation_height_factor,
                           unsigned char zero_byte,
                           const size_t patch_begin_coord[3],
                           size_t patch_elements);
```

#### **`skl_im2row_hwc_e8_zve32x`**, **`skl_im2row_hwc_e16_zve32x`**, **`skl_im2row_hwc_e32_zve32x`**
Type-safe versions of `skl_im2row_hwc_zve32x` for 8-bit, 16-bit, and 32-bit element types respectively.

```c
void skl_im2row_hwc_e8_zve32x(uint8_t *out, const uint8_t *in_batch,
                              int32_t in_w_origin, int32_t in_h_origin,
                              size_t input_height, size_t input_width,
                              size_t input_channel, size_t filter_height,
                              size_t filter_width, size_t dilation_width_factor,
                              size_t dilation_height_factor,
                              unsigned char zero_byte,
                              const size_t patch_begin_coord[3],
                              size_t patch_elements);

void skl_im2row_hwc_e16_zve32x(uint16_t *out, const uint16_t *in_batch,
                               int32_t in_w_origin, int32_t in_h_origin,
                               size_t input_height, size_t input_width,
                               size_t input_channel, size_t filter_height,
                               size_t filter_width, size_t dilation_width_factor,
                               size_t dilation_height_factor,
                               unsigned char zero_byte,
                               const size_t patch_begin_coord[3],
                               size_t patch_elements);

void skl_im2row_hwc_e32_zve32x(uint32_t *out, const uint32_t *in_batch,
                               int32_t in_w_origin, int32_t in_h_origin,
                               size_t input_height, size_t input_width,
                               size_t input_channel, size_t filter_height,
                               size_t filter_width, size_t dilation_width_factor,
                               size_t dilation_height_factor,
                               unsigned char zero_byte,
                               const size_t patch_begin_coord[3],
                               size_t patch_elements);
```

