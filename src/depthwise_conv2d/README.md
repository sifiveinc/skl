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

### Scalar Implementation (Int8)
#### **`skl_depthwise_conv2d_hwc_i8_i8_i32_ref`**
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_hwc_i8_i8_i32_ref(
    int32_t *output,                   // Output tensor (HWC layout)
    const int8_t *input,               // Input tensor (HWC layout)
    const int8_t *filter,              // Filter tensor (HWIM layout)
    size_t input_height,               // Input height dimension
    size_t input_width,                // Input width dimension
    size_t input_channel,              // Input channel dimension
    size_t filter_height,              // Filter height dimension
    size_t filter_width,               // Filter width dimension
    size_t output_height,              // Output height dimension
    size_t output_width,               // Output width dimension
    size_t output_channel,             // Output channel dimension
    size_t depth_multiplier,           // Number of filters per input channel
    size_t stride_height,              // Vertical stride
    size_t stride_width,               // Horizontal stride
    size_t dilation_height_factor,     // Vertical dilation factor
    size_t dilation_width_factor,      // Horizontal dilation factor
    size_t input_row_stride,           // Input row stride in elements
    size_t input_col_stride,           // Input column stride in elements
    size_t filter_row_stride,          // Filter row stride in elements
    size_t filter_col_stride,          // Filter column stride in elements
    size_t output_row_stride,          // Output row stride in elements
    size_t output_col_stride,          // Output column stride in elements
    int32_t input_zero_point           // Input zero-point for quantization
);
```


### Scalar Implementation (Float16)
#### `skl_depthwise_conv2d_hwc_f16_f16_f16_ref`
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_hwc_f16_f16_f16_ref(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

### Scalar Implementation (Float32)
#### `skl_depthwise_conv2d_hwc_f32_f32_f32_ref`
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_hwc_f32_f32_f32_ref(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

### Naming Rule for RVV Implementation

Below is the format of RVV Kernel name:

**`skl_depthwise_conv2d_v[dimensions]_f[number]x[number]_s[number]_d[number]_i[number]_[layout]_[data_type]_[architecture]`**

where
- `v` for vectorized dimensions. `[dimension]` should be a combination of `h` (height), `w` (width) and `c` (channel)
- `f` for filter size. `[number]` should be an exact positive integer or `n` for any positive integer
- `s` for strides
- `d` for dilations
- `i` for input channel size
- `layout` for input/output data layout, `hwc` or `chw`
- `data_type` for data type of tensors (input, filter, output) and separated by underscores
- `architecture` for target architecture

### RVV Implementations (Int8)
#### **`skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_i8_i8_i32_zve32x`**
- Generic RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_i8_i8_i32_zve32x(
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

#### **`skl_depthwise_conv2d_vc_f3x3_sn_dn_m1_in_hwc_i8_i8_i32_zve32x`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters and `depth_multiplier` = 1
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_f3x3_sn_dn_m1_in_hwc_i8_i8_i32_zve32x(
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
#### **`skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f16_f16_f16_zvfh`**
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f16_f16_f16_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

#### **`skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f16_f16_f16_zvfh`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f16_f16_f16_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t depth_multiplier, size_t stride_height, size_t stride_width,
    size_t dilation_height_factor, size_t dilation_width_factor,
    size_t input_row_stride, size_t input_col_stride, size_t filter_row_stride,
    size_t filter_col_stride, size_t output_row_stride, size_t output_col_stride);
```

### RVV Implementations (Float32)
#### **`skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f32_f32_f32_zve32f`**
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f32_f32_f32_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

#### **`skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f32_f32_f32_zve32f`**
- Specialized RVV implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)
- Optimized for 3x3 filters
- Vectorize along the channel dimension

```c
void skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f32_f32_f32_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```
