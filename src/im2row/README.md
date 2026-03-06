# Im2Row Kernels

The `im2row` directory contains functions for converting HWC layout input tensors into im2row matrices, enabling efficient convolution-to-GEMM transformation. These functions process **all output positions** for a single batch, generating the complete im2row matrix in one call.

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

## API Design

The im2row kernels process **entire batches** in a single call, generating the complete im2row matrix for all output positions. The API takes standard convolution parameters (`stride`, `padding`, `dilation`, `output_height`, `output_width`) rather than individual patch coordinates.

### Key Parameters

- **`output`**: Output im2row matrix buffer (size: `output_height × output_width × filter_height × filter_width × input_channel`)
- **`input`**: Input tensor for single batch (HWC layout)
- **`input_height_stride`**, **`input_width_stride`**: Strides for navigating input tensor (for standard HWC: `width_stride = input_channel`, `height_stride = input_width × input_channel`)
- **`filter_height`**, **`filter_width`**: Convolution filter dimensions
- **`output_height`**, **`output_width`**: Output tensor dimensions (determines number of patches to extract)
- **`padding_width`**, **`padding_height`**: Padding applied to input
- **`stride_width`**, **`stride_height`**: Convolution stride
- **`dilation_width`**, **`dilation_height`**: Dilation factors
- **`zero_byte`**: Byte value for padding (typically 0x00)

### Optimization

The RVV implementations automatically select specialized non-dilated kernels when `dilation_width == 1 && dilation_height == 1` for better performance.

## Kernel List

### Scalar Implementations

#### **`skl_im2row_generic_hwc`**
Processes all output positions for a single batch using element-by-element processing. Supports any primitive data type.

```c
void skl_im2row_generic_hwc(
    void *output, const void *input, size_t element_size, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);
```

**Parameters:**
- `output` - Output im2row matrix buffer (size: `output_height × output_width × filter_height × filter_width × input_channel`)
- `input` - Input tensor for single batch (HWC layout)
- `element_size` - Size in bytes of element type (e.g., `sizeof(float)`)
- `input_height`, `input_width`, `input_channel` - Input tensor dimensions
- `input_height_stride`, `input_width_stride` - Strides for navigating input tensor
- `filter_height`, `filter_width` - Convolution filter dimensions
- `output_height`, `output_width` - Output tensor dimensions
- `padding_width`, `padding_height` - Padding applied to input
- `stride_width`, `stride_height` - Convolution stride
- `dilation_width`, `dilation_height` - Dilation factors
- `zero_byte` - Byte value for padding (typically 0x00)

### RVV Implementations

#### **`skl_im2row_hwc_zve32x`**
Processes all output positions for a single batch using optimized bulk memory operations. Supports any primitive data type. Automatically selects the specialized non-dilated kernel when `dilation_width == 1 && dilation_height == 1`.

```c
void skl_im2row_hwc_zve32x(
    void *output, const void *input, size_t element_size, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);
```

**Parameters:** Same as `skl_im2row_generic_hwc`

#### **`skl_im2row_hwc_e8_zve32x`**, **`skl_im2row_hwc_e16_zve32x`**, **`skl_im2row_hwc_e32_zve32x`**
Type-safe versions of `skl_im2row_hwc_zve32x` for 8-bit, 16-bit, and 32-bit element types respectively.

```c
void skl_im2row_hwc_e8_zve32x(
    uint8_t *output, const uint8_t *input, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);

void skl_im2row_hwc_e16_zve32x(
    uint16_t *output, const uint16_t *input, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);

void skl_im2row_hwc_e32_zve32x(
    uint32_t *output, const uint32_t *input, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);
```

**Parameters:** Same as `skl_im2row_hwc_zve32x`, but with typed pointers instead of `void*` and no `element_size` parameter.

