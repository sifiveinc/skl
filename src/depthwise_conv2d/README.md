# Depthwise Conv2D Kernels

This directory contains optimized kernels for Depthwise Convolution 2D operations. Depthwise convolution is a specialized type of convolution where each input channel is convolved with a single, dedicated filter, rather than combining information across all channels as in standard convolution. This operation significantly reduces computational cost and model parameters, making it a key component in efficient neural network architectures like MobileNets.

Currently supports `int8` input/filter with `int32` output, as well as `float16` and `float32`.

## Constraints

### No Padding Supported
- Padding should be handled separately if required
- Input and output dimensions must be calculated considering stride and dilation without padding

### No Requantization
- Post-processing steps like bias addition or requantization to `int8` must be handled separately

### Zero-point of Filter Must Be 0
- All filter weights must have a zero-point of 0 (symmetric quantization)
- Input zero-point is supported and passed as a parameter

## Kernel List

### Naming Rule for RVV Implementation

Below is the format of RVV Kernel name:

**`skl_depthwise_conv2d_<specialization>_<input_data_type><input_layout>_<filter_data_type><filter_layout>_<output_data_type><output_layout>_<isa>`**

where
- `specialization` includes the contraints of filter size, strides, and dilations, expressed in `f[number]x[number]s[number]d[number]i[number]`.
If any of them is not specified, it indicates any positive integer.
  - `f` for filter size. `[number]` should be an exact positive integer
  - `s` for strides
  - `d` for dilations
  - `i` for input channel size
- `layout`: `hwc` or `chw` for input/output data layout. `hwim` for filter data layout.
- `data_type` for data type of tensors (input, filter, output).
- `isa` for target architecture, e.g. `zve32x`.

Below is the description of filter layout:
```
Given M = depth multiplier > 1, I = input channel size
In general, filter's channel size is equal to M * input's channel size.

Here are the pairs of input and filter for channel-wise dot-product, where input's elements will be repeated M times in each channel group.

Input channel index:  [0, 0, ..., 0]     [1, 1,     ..., 1]         [(I - 1),     (I - 1),         ..., (I - 1)]
Filter channel index: [0, 1, ..., M - 1] [M, M + 1, ..., 2 * M - 1] [(I - 1) * M, (I - 1) * M + 1, ..., I * M - 1]

In the special case M = 1, filter's channel size is equal to input's channel size.

The pairs will be iterated through the height and width of filter tensor for each output element.
Following the order of nested loops, we express filter as `hwim` from the outmost loop to the innermost loop.
```

### RVV Implementations (Int8)
#### **`skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_zve32x`**
- Generic RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_zve32x(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point);
```

#### **`skl_depthwise_conv2d_f3x3m1_i8hwc_i8hwim_i32hwc_zve32x`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters and `depth_multiplier` = 1

```c
void skl_depthwise_conv2d_f3x3m1_i8hwc_i8hwim_i32hwc_zve32x(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point);
```

### RVV Implementations (Float16)
#### **`skl_depthwise_conv2d_f16hwc_f16hwim_f16hwc_zvfh`**
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_f16hwc_f16hwim_f16hwc_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

#### **`skl_depthwise_conv2d_f3x3_f16hwc_f16hwim_f16hwc_zvfh`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters

```c
void skl_depthwise_conv2d_f3x3_f16hwc_f16hwim_f16hwc_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t depth_multiplier, size_t stride_height, size_t stride_width,
    size_t dilation_height_factor, size_t dilation_width_factor,
    size_t input_row_stride, size_t input_col_stride, size_t filter_row_stride,
    size_t filter_col_stride, size_t output_row_stride, size_t output_col_stride);
```

### RVV Implementations (Float32)
#### **`skl_depthwise_conv2d_f32hwc_f32hwim_f32hwc_zve32f`**
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_f32hwc_f32hwim_f32hwc_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

#### **`skl_depthwise_conv2d_f3x3_f32hwc_f32hwim_f32hwc_zve32f`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters

```c
void skl_depthwise_conv2d_f3x3_f32hwc_f32hwim_f32hwc_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```
