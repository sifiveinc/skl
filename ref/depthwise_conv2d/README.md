# Reference Depthwise Conv2D Implementations

This directory contains reference kernels for Depthwise Convolution 2D operations.
These are scalar implementations used for correctness verification and as reference for optimized ISA-specific implementations.

Currently supports `int8` input/filter with `int32` output, as well as `float16`, `float32` and `float64`.

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
#### **`skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_ref`**
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_ref(
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
#### `skl_depthwise_conv2d_f16hwc_f16hwim_f16hwc_ref`
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_f16hwc_f16hwim_f16hwc_ref(
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
#### `skl_depthwise_conv2d_f32hwc_f32hwim_f32hwc_ref`
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_f32hwc_f32hwim_f32hwc_ref(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```

### Scalar Implementation (Float64)
#### `skl_depthwise_conv2d_f64hwc_f64hwim_f64hwc_ref`
- Generic implementation
- Data layout: Input (HWC), Filter (HWIM), Output (HWC)

```c
void skl_depthwise_conv2d_f64hwc_f64hwim_f64hwc_ref(
    double *output, const double *input, const double *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);
```
