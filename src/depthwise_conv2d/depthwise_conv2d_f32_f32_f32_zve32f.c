// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

__attribute__((unused)) SKL_FUNC_PRIVATE vfloat32m4_t
skl_depthwise_conv2d_post_process_output_clamp_vf32m4(
    vfloat32m4_t v_sum, size_t vl, float output_activation_min,
    float output_activation_max) {
  v_sum = __riscv_vfmax_vf_f32m4(v_sum, output_activation_min, vl);
  v_sum = __riscv_vfmin_vf_f32m4(v_sum, output_activation_max, vl);
  return v_sum;
}

SKL_FUNC_PRIVATE vfloat32m4_t
skl_depthwise_conv2d_post_process_output_linear_vf32m4(
    vfloat32m4_t v_sum, size_t vl, float output_activation_min,
    float output_activation_max) {
  (void)vl;
  (void)output_activation_min;
  (void)output_activation_max;
  return v_sum;
}

SKL_FUNC_PRIVATE void
skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f32_f32_f32_internal_zve32f(
    float *output, const float *input, const float *filter, const float *bias,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t pad_height, size_t pad_width, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    float output_activation_min, float output_activation_max) {
  size_t output_next_row_remaining_offset =
      output_row_stride - output_width * output_col_stride;

  if (depth_multiplier == 1) {
    for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
         output_c_avl -= vl) {
      vl = __riscv_vsetvl_e32m4(output_c_avl);

      const float *input_ptr = input;
      float *output_ptr = output;

      // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
      vfloat32m4_t v_init_output_f32 = __riscv_vundefined_f32m4();
      // NOLINTEND(clang-analyzer-deadcode.DeadStores)
      if (bias != NULL) {
        v_init_output_f32 = __riscv_vle32_v_f32m4(bias, vl);
      } else {
        v_init_output_f32 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      }

      for (size_t output_h = 0; output_h < output_height; output_h++) {
        for (size_t output_w = 0; output_w < output_width; output_w++) {
          vfloat32m4_t v_output_f32 =
              __riscv_vmv_v_v_f32m4(v_init_output_f32, vl);

          int32_t in_y =
              -(int32_t)pad_height + (int32_t)output_h * (int32_t)stride_height;
          for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
            if (0 <= in_y && in_y < (int32_t)input_height) {
              int32_t in_x = -(int32_t)pad_width +
                             (int32_t)output_w * (int32_t)stride_width;
              for (size_t filter_w = 0; filter_w < filter_width; filter_w++) {
                if (0 <= in_x && in_x < (int32_t)input_width) {
                  vfloat32m4_t v_input_f32 = __riscv_vle32_v_f32m4(
                      input_ptr + (size_t)in_y * input_row_stride +
                          (size_t)in_x * input_col_stride,
                      vl);
                  vfloat32m4_t v_filter_f32 = __riscv_vle32_v_f32m4(
                      filter + filter_h * filter_row_stride +
                          filter_w * filter_col_stride,
                      vl);

                  v_output_f32 = __riscv_vfmacc_vv_f32m4(
                      v_output_f32, v_input_f32, v_filter_f32, vl);
                }
                in_x += (int32_t)dilation_width_factor;
              }
            }
            in_y += (int32_t)dilation_height_factor;
          }

          post_process_output_func(v_output_f32, vl, output_activation_min,
                                   output_activation_max);
          __riscv_vse32_v_f32m4(output_ptr, v_output_f32, vl);

          output_ptr += output_col_stride;
        }

        output_ptr += output_next_row_remaining_offset;
      }
      input += vl;
      filter += vl;
      if (bias != NULL) {
        bias += vl;
      }
      output += vl;
    }
  } else {
    if (input_channel == 1) {
      for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
           output_c_avl -= vl) {
        vl = __riscv_vsetvl_e32m4(output_c_avl);

        const float *input_ptr = input;
        float *output_ptr = output;

        // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
        vfloat32m4_t v_init_output_f32 = __riscv_vundefined_f32m4();
        // NOLINTEND(clang-analyzer-deadcode.DeadStores)
        if (bias != NULL) {
          v_init_output_f32 = __riscv_vle32_v_f32m4(bias, vl);
        } else {
          v_init_output_f32 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }

        for (size_t output_h = 0; output_h < output_height; output_h++) {
          for (size_t output_w = 0; output_w < output_width; output_w++) {
            vfloat32m4_t v_output_f32 =
                __riscv_vmv_v_v_f32m4(v_init_output_f32, vl);

            int32_t in_y = -(int32_t)pad_height +
                           (int32_t)output_h * (int32_t)stride_height;
            for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
              if (0 <= in_y && in_y < (int32_t)input_height) {
                int32_t in_x = -(int32_t)pad_width +
                               (int32_t)output_w * (int32_t)stride_width;
                for (size_t filter_w = 0; filter_w < filter_width; filter_w++) {
                  if (0 <= in_x && in_x < (int32_t)input_width) {
                    vfloat32m4_t v_input_f32 = __riscv_vlse32_v_f32m4(
                        input_ptr + (size_t)in_y * input_row_stride +
                            (size_t)in_x * input_col_stride,
                        0, vl);
                    vfloat32m4_t v_filter_f32 = __riscv_vle32_v_f32m4(
                        filter + filter_h * filter_row_stride +
                            filter_w * filter_col_stride,
                        vl);

                    v_output_f32 = __riscv_vfmacc_vv_f32m4(
                        v_output_f32, v_input_f32, v_filter_f32, vl);
                  }
                  in_x += (int32_t)dilation_width_factor;
                }
              }
              in_y += (int32_t)dilation_height_factor;
            }

            post_process_output_func(v_output_f32, vl, output_activation_min,
                                     output_activation_max);
            __riscv_vse32_v_f32m4(output_ptr, v_output_f32, vl);

            output_ptr += output_col_stride;
          }

          output_ptr += output_next_row_remaining_offset;
        }
        filter += vl;
        if (bias != NULL) {
          bias += vl;
        }
        output += vl;
      }
    } else {
      size_t float_data_stride = depth_multiplier * sizeof(float);

      for (size_t input_c_avl = input_channel, vl; input_c_avl > 0;
           input_c_avl -= vl) {
        vl = __riscv_vsetvl_e32m4(input_c_avl);

        for (size_t m = 0; m < depth_multiplier; m++) {

          const float *input_ptr = input;
          float *output_ptr = output;

          // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
          vfloat32m4_t v_init_output_f32 = __riscv_vundefined_f32m4();
          // NOLINTEND(clang-analyzer-deadcode.DeadStores)
          if (bias != NULL) {
            v_init_output_f32 =
                __riscv_vlse32_v_f32m4(bias, (ptrdiff_t)float_data_stride, vl);
          } else {
            v_init_output_f32 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
          }

          for (size_t output_h = 0; output_h < output_height; output_h++) {
            for (size_t output_w = 0; output_w < output_width; output_w++) {
              vfloat32m4_t v_output_f32 =
                  __riscv_vmv_v_v_f32m4(v_init_output_f32, vl);

              int32_t in_y = -(int32_t)pad_height +
                             (int32_t)output_h * (int32_t)stride_height;
              for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
                if (0 <= in_y && in_y < (int32_t)input_height) {
                  int32_t in_x = -(int32_t)pad_width +
                                 (int32_t)output_w * (int32_t)stride_width;
                  for (size_t filter_w = 0; filter_w < filter_width;
                       filter_w++) {
                    if (0 <= in_x && in_x < (int32_t)input_width) {
                      vfloat32m4_t v_input_f32 = __riscv_vle32_v_f32m4(
                          input_ptr + (size_t)in_y * input_row_stride +
                              (size_t)in_x * input_col_stride,
                          vl);
                      vfloat32m4_t v_filter_f32 = __riscv_vlse32_v_f32m4(
                          filter + filter_h * filter_row_stride +
                              filter_w * filter_col_stride,
                          (ptrdiff_t)float_data_stride, vl);

                      v_output_f32 = __riscv_vfmacc_vv_f32m4(
                          v_output_f32, v_input_f32, v_filter_f32, vl);
                    }
                    in_x += (int32_t)dilation_width_factor;
                  }
                }
                in_y += (int32_t)dilation_height_factor;
              }

              post_process_output_func(v_output_f32, vl, output_activation_min,
                                       output_activation_max);
              __riscv_vsse32_v_f32m4(output_ptr, (ptrdiff_t)float_data_stride,
                                     v_output_f32, vl);

              output_ptr += output_col_stride;
            }

            output_ptr += output_next_row_remaining_offset;
          }

          filter++;
          if (bias != NULL) {
            bias++;
          }
          output++;
        }

        input += vl;
        filter += (vl - 1) * depth_multiplier;
        if (bias != NULL) {
          bias += (vl - 1) * depth_multiplier;
        }
        output += (vl - 1) * depth_multiplier;
      }
    }
  }
}

SKL_FUNC void skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f32_f32_f32_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t output_channel, size_t depth_multiplier, size_t stride_height,
    size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride) {
  skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f32_f32_f32_internal_zve32f(
      output, input, filter, NULL, input_height, input_width, input_channel,
      filter_height, filter_width, output_height, output_width, output_channel,
      depth_multiplier, 0, 0, stride_height, stride_width,
      dilation_height_factor, dilation_width_factor, input_row_stride,
      input_col_stride, filter_row_stride, filter_col_stride, output_row_stride,
      output_col_stride, skl_depthwise_conv2d_post_process_output_linear_vf32m4,
      0.0f, 0.0f);
}

SKL_FUNC_PRIVATE vfloat32m4_t skl_depthwise_conv2d_unit_stride_load_vf32m4(
    const float *ptr, size_t data_stride, size_t vl) {
  (void)data_stride;
  return __riscv_vle32_v_f32m4(ptr, vl);
}

SKL_FUNC_PRIVATE vfloat32m4_t skl_depthwise_conv2d_strided_load_vf32m4(
    const float *ptr, size_t data_stride, size_t vl) {
  return __riscv_vlse32_v_f32m4(ptr, (ptrdiff_t)data_stride, vl);
}

SKL_FUNC_PRIVATE void
skl_depthwise_conv2d_unit_stride_store_vf32m4(float *ptr, size_t data_stride,
                                              vfloat32m4_t v_data, size_t vl) {
  (void)data_stride;
  __riscv_vse32_v_f32m4(ptr, v_data, vl);
}

SKL_FUNC_PRIVATE void
skl_depthwise_conv2d_strided_store_vf32m4(float *ptr, size_t data_stride,
                                          vfloat32m4_t v_data, size_t vl) {
  __riscv_vsse32_v_f32m4(ptr, (ptrdiff_t)data_stride, v_data, vl);
}

SKL_FUNC_PRIVATE vfloat32m4_t skl_depthwise_conv2d_init_accum_by_vector_vf32m4(
    vfloat32m4_t v_init_sum, size_t vl, const float *output,
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    size_t data_stride) {
  (void)output;
  (void)load_func;
  (void)data_stride;
  return __riscv_vmv_v_v_f32m4(v_init_sum, vl);
}

SKL_FUNC_PRIVATE vfloat32m4_t
skl_depthwise_conv2d_init_accum_by_partial_accum_vf32m4(
    vfloat32m4_t v_init_sum, size_t vl, const float *output,
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    size_t data_stride) {
  (void)v_init_sum;
  return load_func(output, data_stride, vl);
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_padding_sliding_vf32m4(
    vfloat32m4_t v_init_sum, float *output, size_t out_x_start,
    size_t out_x_end, size_t output_col_stride, float output_activation_min,
    float output_activation_max, size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  for (size_t out_x = out_x_start; out_x < out_x_end; out_x++) {
    vfloat32m4_t v_sum =
        init_sum_func(v_init_sum, vl, output + out_x * output_col_stride,
                      load_func, data_stride);
    v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                     output_activation_max);
    store_func(output + out_x * output_col_stride, data_stride, v_sum, vl);
  }
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_overlapped_0_sliding_vf32m4(
    const float *input, vfloat32m4_t v_filter_0, vfloat32m4_t v_filter_1,
    vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum, float *output,
    size_t input_width, size_t output_width, size_t pad_width,
    size_t stride_width, size_t dilation_width_factor, size_t input_col_stride,
    size_t output_col_stride, float output_activation_min,
    float output_activation_max, size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  int32_t left_in_x = -(int32_t)pad_width;
  int32_t mid_in_x = left_in_x + (int32_t)dilation_width_factor;
  int32_t right_in_x = mid_in_x + (int32_t)dilation_width_factor;
  for (size_t out_x = 0; out_x < output_width; out_x++) {
    vfloat32m4_t v_sum =
        init_sum_func(v_init_sum, vl, output + out_x * output_col_stride,
                      load_func, data_stride);
    if (left_in_x >= 0 && left_in_x < (int32_t)input_width) {
      vfloat32m4_t v_left_input =
          __riscv_vle32_v_f32m4(input + left_in_x * input_col_stride, vl);
      v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_left_input, v_filter_0, vl);
    }
    if (mid_in_x >= 0 && mid_in_x < (int32_t)input_width) {
      vfloat32m4_t v_mid_input =
          __riscv_vle32_v_f32m4(input + mid_in_x * input_col_stride, vl);
      v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_mid_input, v_filter_1, vl);
    }
    if (right_in_x >= 0 && right_in_x < (int32_t)input_width) {
      vfloat32m4_t v_right_input =
          __riscv_vle32_v_f32m4(input + right_in_x * input_col_stride, vl);
      v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_right_input, v_filter_2, vl);
    }
    v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                     output_activation_max);
    store_func(output + out_x * output_col_stride, data_stride, v_sum, vl);

    left_in_x += (int32_t)stride_width;
    mid_in_x += (int32_t)stride_width;
    right_in_x += (int32_t)stride_width;
  }
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_dot_product_1x3_vf32m4(
    vfloat32m4_t v_left_input, vfloat32m4_t v_mid_input,
    vfloat32m4_t v_right_input, vfloat32m4_t v_filter_0,
    vfloat32m4_t v_filter_1, vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum,
    float *output, float output_activation_min, float output_activation_max,
    size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  vfloat32m4_t v_sum =
      init_sum_func(v_init_sum, vl, output, load_func, data_stride);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_left_input, v_filter_0, vl);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_mid_input, v_filter_1, vl);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_right_input, v_filter_2, vl);
  v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                   output_activation_max);
  store_func(output, data_stride, v_sum, vl);
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_dot_product_1x3_partial_001_vf32m4(
    vfloat32m4_t v_left_input, vfloat32m4_t v_mid_input,
    vfloat32m4_t v_right_input, vfloat32m4_t v_filter_0,
    vfloat32m4_t v_filter_1, vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum,
    float *output, float output_activation_min, float output_activation_max,
    size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  (void)v_left_input;
  (void)v_mid_input;
  (void)v_filter_0;
  (void)v_filter_1;
  vfloat32m4_t v_sum =
      init_sum_func(v_init_sum, vl, output, load_func, data_stride);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_right_input, v_filter_2, vl);
  v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                   output_activation_max);
  store_func(output, data_stride, v_sum, vl);
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_dot_product_1x3_partial_011_vf32m4(
    vfloat32m4_t v_left_input, vfloat32m4_t v_mid_input,
    vfloat32m4_t v_right_input, vfloat32m4_t v_filter_0,
    vfloat32m4_t v_filter_1, vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum,
    float *output, float output_activation_min, float output_activation_max,
    size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  (void)v_left_input;
  (void)v_filter_0;
  vfloat32m4_t v_sum =
      init_sum_func(v_init_sum, vl, output, load_func, data_stride);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_mid_input, v_filter_1, vl);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_right_input, v_filter_2, vl);
  v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                   output_activation_max);
  store_func(output, data_stride, v_sum, vl);
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
    vfloat32m4_t v_left_input, vfloat32m4_t v_mid_input,
    vfloat32m4_t v_right_input, vfloat32m4_t v_filter_0,
    vfloat32m4_t v_filter_1, vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum,
    float *output, float output_activation_min, float output_activation_max,
    size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  (void)v_mid_input;
  (void)v_right_input;
  (void)v_filter_1;
  (void)v_filter_2;
  vfloat32m4_t v_sum =
      init_sum_func(v_init_sum, vl, output, load_func, data_stride);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_left_input, v_filter_0, vl);
  v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                   output_activation_max);
  store_func(output, data_stride, v_sum, vl);
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
    vfloat32m4_t v_left_input, vfloat32m4_t v_mid_input,
    vfloat32m4_t v_right_input, vfloat32m4_t v_filter_0,
    vfloat32m4_t v_filter_1, vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum,
    float *output, float output_activation_min, float output_activation_max,
    size_t vl, size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  (void)v_right_input;
  (void)v_filter_2;
  vfloat32m4_t v_sum =
      init_sum_func(v_init_sum, vl, output, load_func, data_stride);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_left_input, v_filter_0, vl);
  v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_mid_input, v_filter_1, vl);
  v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                   output_activation_max);
  store_func(output, data_stride, v_sum, vl);
}

// Slide 1x3 filter on a row where:
//   (1) Overlapping happens
//   (2) Area is large enough for one overlap period
SKL_FUNC_PRIVATE void skl_depthwise_conv2d_overlapped_1_sliding_vf32m4(
    const float *input, vfloat32m4_t v_filter_0, vfloat32m4_t v_filter_1,
    vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum, float *output,
    size_t input_width, size_t output_width, size_t pad_width,
    size_t stride_width, size_t dilation_width_factor, size_t input_col_stride,
    size_t output_col_stride, float output_activation_min,
    float output_activation_max, size_t overlap_period,
    size_t non_pad_out_x_start, size_t non_pad_out_x_end, size_t vl,
    size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  size_t next_input_stride = dilation_width_factor * input_col_stride;
  size_t next_output_stride = overlap_period * output_col_stride;

  for (size_t out_start_offset = 0; out_start_offset < overlap_period;
       out_start_offset++) {
    //        (padding)      (non-padding)
    //         .  .  .  O  O  O  O  O  O  O  ......
    // out_x:           I0 I1 I2
    //                  ^
    //                  current_input
    size_t out_x = non_pad_out_x_start + out_start_offset;
    const float *current_input =
        input + ((out_x * stride_width) - pad_width) * input_col_stride;
    float *current_output = output + out_x * output_col_stride;

    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vfloat32m4_t v_input_0 = __riscv_vundefined_f32m4();
    vfloat32m4_t v_input_1 = __riscv_vundefined_f32m4();
    vfloat32m4_t v_input_2 = __riscv_vundefined_f32m4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
    current_input += next_input_stride;

    // cross-padding
    if ((int32_t)out_x - (int32_t)overlap_period >= 0) {
      if ((int32_t)out_x * (int32_t)stride_width - (int32_t)pad_width -
              (int32_t)dilation_width_factor >=
          0) {
        //                           .  .  O  O  O  O  O
        // out_x - overlap_period:      -  I2 I0
        // out_x                 :            I0 I1 I2
        //                                       ^
        //                                       current_input
        v_input_2 = __riscv_vle32_v_f32m4(
            current_input - next_input_stride - next_input_stride, vl);

        skl_depthwise_conv2d_dot_product_1x3_partial_011_vf32m4(
            __riscv_vundefined_f32m4(), v_input_2, v_input_0, v_filter_0,
            v_filter_1, v_filter_2, v_init_sum,
            current_output - next_output_stride, output_activation_min,
            output_activation_max, vl, data_stride, init_sum_func,
            post_process_output_func, load_func, store_func);
      } else {
        //                           .  .  .  O  O  O  O
        // out_x - overlap_period:      -  -  I0
        // out_x                 :            I0 I1 I2
        //                                       ^
        //                                       current_input
        skl_depthwise_conv2d_dot_product_1x3_partial_001_vf32m4(
            __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(), v_input_0,
            v_filter_0, v_filter_1, v_filter_2, v_init_sum,
            current_output - next_output_stride, output_activation_min,
            output_activation_max, vl, data_stride, init_sum_func,
            post_process_output_func, load_func, store_func);
      }
    }

    // non-padding
    while (out_x + 2 * overlap_period < non_pad_out_x_end) {
      v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_0, v_input_1, v_input_2, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_2, v_input_0, v_input_1, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_1, v_input_2, v_input_0, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      out_x += 3 * overlap_period;
    }

    // remaining outputs after unroll loop
    size_t counter = 0;
    if (out_x < non_pad_out_x_end) {
      v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_0, v_input_1, v_input_2, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;
      out_x += overlap_period;
      counter++;

      if (out_x < non_pad_out_x_end) {
        v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
        current_input += next_input_stride;
        v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
        current_input += next_input_stride;
        skl_depthwise_conv2d_dot_product_1x3_vf32m4(
            v_input_2, v_input_0, v_input_1, v_filter_0, v_filter_1, v_filter_2,
            v_init_sum, current_output, output_activation_min,
            output_activation_max, vl, data_stride, init_sum_func,
            post_process_output_func, load_func, store_func);
        current_output += next_output_stride;
        out_x += overlap_period;
        counter++;
      }
    }

    // cross padding
    if (out_x < output_width) {
      switch (counter) {
      case 0:
        if (out_x * stride_width - pad_width + dilation_width_factor <
            input_width) {
          //                           O  O  O  O  O  .  .
          // out_x - overlap_period:      I1 I2 I0
          // out_x                 :            I0 I1 -
          v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
          skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
              v_input_0, v_input_1, __riscv_vundefined_f32m4(), v_filter_0,
              v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        } else {
          //                           O  O  O  O  .  .  .
          // out_x - overlap_period:      I1 I2 I0
          // out_x                 :            I0 -  -
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_0, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      case 1:
        if (out_x * stride_width - pad_width + dilation_width_factor <
            input_width) {
          //                           O  O  O  O  O  .  .
          // out_x - overlap_period:      I0 I1 I2
          // out_x                 :            I2 I0 -
          v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
          skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
              v_input_2, v_input_0, __riscv_vundefined_f32m4(), v_filter_0,
              v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        } else {
          //                           O  O  O  O  .  .  .
          // out_x - overlap_period:      I0 I1 I2
          // out_x                 :            I2 -  -
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_2, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      case 2:
        if (out_x * stride_width - pad_width + dilation_width_factor <
            input_width) {
          //                           O  O  O  O  O  .  .
          // out_x - overlap_period:      I2 I0 I1
          // out_x                 :            I1 I2 -
          v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
          skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
              v_input_1, v_input_2, __riscv_vundefined_f32m4(), v_filter_0,
              v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        } else {
          //                           O  O  O  O  .  .  .
          // out_x - overlap_period:      I2 I0 I1
          // out_x                 :            I1 -  -
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_1, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      default:
        break;
      }
    }
  }
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_overlapped_2_sliding_vf32m4(
    const float *input, vfloat32m4_t v_filter_0, vfloat32m4_t v_filter_1,
    vfloat32m4_t v_filter_2, vfloat32m4_t v_init_sum, float *output,
    size_t output_width, size_t pad_width, size_t stride_width,
    size_t dilation_width_factor, size_t input_col_stride,
    size_t output_col_stride, float output_activation_min,
    float output_activation_max, size_t overlap_period,
    size_t non_pad_out_x_start, size_t non_pad_out_x_end, size_t vl,
    size_t data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  size_t next_input_stride = dilation_width_factor * input_col_stride;
  size_t next_output_stride = overlap_period * output_col_stride;

  for (size_t out_start_offset = 0; out_start_offset < overlap_period;
       out_start_offset++) {
    //        (padding)      (non-padding)
    //         .  .  .  O  O  O  O  O  O  O  ......
    // out_x:           I0 I1 I2
    //                  ^
    //                  current_input
    size_t out_x = non_pad_out_x_start + out_start_offset;
    const float *current_input =
        input + ((out_x * stride_width) - pad_width) * input_col_stride;
    float *current_output = output + out_x * output_col_stride;

    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vfloat32m4_t v_input_0 = __riscv_vundefined_f32m4();
    vfloat32m4_t v_input_1 = __riscv_vundefined_f32m4();
    vfloat32m4_t v_input_2 = __riscv_vundefined_f32m4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
    current_input += next_input_stride;
    v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
    current_input += next_input_stride;

    // cross-padding
    //                              .  .  .  O  O  O  O
    // out_x - 2 * overlap_period:     -  -  I0         (if exist)
    // out_x - overlap_period    :        -  I0 I1      (if exist)
    // out_x                     :           I0 I1 I2
    if ((int32_t)out_x - 2 * (int32_t)overlap_period >= 0) {
      skl_depthwise_conv2d_dot_product_1x3_partial_001_vf32m4(
          __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(), v_input_0,
          v_filter_0, v_filter_1, v_filter_2, v_init_sum,
          current_output - 2 * next_output_stride, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
    }

    if ((int32_t)out_x - (int32_t)overlap_period >= 0) {
      skl_depthwise_conv2d_dot_product_1x3_partial_011_vf32m4(
          __riscv_vundefined_f32m4(), v_input_0, v_input_1, v_filter_0,
          v_filter_1, v_filter_2, v_init_sum,
          current_output - next_output_stride, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
    }

    // non-padding
    while (out_x + 2 * overlap_period < non_pad_out_x_end) {
      v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_0, v_input_1, v_input_2, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_1, v_input_2, v_input_0, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      v_input_1 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_2, v_input_0, v_input_1, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;

      out_x += 3 * overlap_period;
    }

    // remaining outputs after unroll loop
    size_t counter = 0;
    if (out_x < non_pad_out_x_end) {
      v_input_2 = __riscv_vle32_v_f32m4(current_input, vl);
      current_input += next_input_stride;
      skl_depthwise_conv2d_dot_product_1x3_vf32m4(
          v_input_0, v_input_1, v_input_2, v_filter_0, v_filter_1, v_filter_2,
          v_init_sum, current_output, output_activation_min,
          output_activation_max, vl, data_stride, init_sum_func,
          post_process_output_func, load_func, store_func);
      current_output += next_output_stride;
      out_x += overlap_period;
      counter++;

      if (out_x < non_pad_out_x_end) {
        v_input_0 = __riscv_vle32_v_f32m4(current_input, vl);
        // current_input += next_input_stride;
        skl_depthwise_conv2d_dot_product_1x3_vf32m4(
            v_input_1, v_input_2, v_input_0, v_filter_0, v_filter_1, v_filter_2,
            v_init_sum, current_output, output_activation_min,
            output_activation_max, vl, data_stride, init_sum_func,
            post_process_output_func, load_func, store_func);
        current_output += next_output_stride;
        out_x += overlap_period;
        counter++;
      }
    }

    // cross padding
    if (out_x < output_width) {
      switch (counter) {
      case 0:
        //                           O  O  O  O  .  .  .
        // out_x - overlap_period:      I2 I0 I1
        // out_x                 :         I0 I1 -
        // out_x + overlap_period:            I1 -  -    (if exist)
        skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
            v_input_0, v_input_1, __riscv_vundefined_f32m4(), v_filter_0,
            v_filter_1, v_filter_2, v_init_sum, current_output,
            output_activation_min, output_activation_max, vl, data_stride,
            init_sum_func, post_process_output_func, load_func, store_func);
        current_output += next_output_stride;
        out_x += overlap_period;

        if (out_x < output_width) {
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_1, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      case 1:
        //                           O  O  O  O  .  .  .
        // out_x - overlap_period:      I0 I1 I2
        // out_x                 :         I1 I2 -
        // out_x + overlap_period:            I2 -  -    (if exist)
        skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
            v_input_1, v_input_2, __riscv_vundefined_f32m4(), v_filter_0,
            v_filter_1, v_filter_2, v_init_sum, current_output,
            output_activation_min, output_activation_max, vl, data_stride,
            init_sum_func, post_process_output_func, load_func, store_func);
        current_output += next_output_stride;
        out_x += overlap_period;

        if (out_x < output_width) {
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_2, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      case 2:
        //                           O  O  O  O  .  .  .
        // out_x - overlap_period:      I1 I2 I0
        // out_x                 :         I2 I0 -
        // out_x + overlap_period:            I0 -  -    (if exist)
        skl_depthwise_conv2d_dot_product_1x3_partial_110_vf32m4(
            v_input_2, v_input_0, __riscv_vundefined_f32m4(), v_filter_0,
            v_filter_1, v_filter_2, v_init_sum, current_output,
            output_activation_min, output_activation_max, vl, data_stride,
            init_sum_func, post_process_output_func, load_func, store_func);
        current_output += next_output_stride;
        out_x += overlap_period;

        if (out_x < output_width) {
          skl_depthwise_conv2d_dot_product_1x3_partial_100_vf32m4(
              v_input_0, __riscv_vundefined_f32m4(), __riscv_vundefined_f32m4(),
              v_filter_0, v_filter_1, v_filter_2, v_init_sum, current_output,
              output_activation_min, output_activation_max, vl, data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }
        break;
      default:
        break;
      }
    }
  }
}

// Slide 1x3 filter on a 2D plane
SKL_FUNC_PRIVATE void skl_depthwise_conv2d_process_1x3_vf32m4(
    const float *input, const float *filter, vfloat32m4_t v_init_sum,
    float *output, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, size_t pad_width,
    size_t stride_height, size_t stride_width, size_t dilation_width_factor,
    size_t input_row_stride, size_t input_col_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    float output_activation_min, float output_activation_max,
    size_t non_pad_out_x_start, size_t non_pad_out_x_end, int32_t in_y_start,
    size_t vl, size_t float_data_stride,
    vfloat32m4_t (*init_sum_func)(vfloat32m4_t, size_t, const float *,
                                  vfloat32m4_t (*)(const float *, size_t,
                                                   size_t),
                                  size_t),
    vfloat32m4_t (*post_process_output_func)(vfloat32m4_t, size_t, float,
                                             float),
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  vfloat32m4_t v_filter_0 =
      load_func(filter + 0 * filter_col_stride, float_data_stride, vl);
  vfloat32m4_t v_filter_1 =
      load_func(filter + 1 * filter_col_stride, float_data_stride, vl);
  vfloat32m4_t v_filter_2 =
      load_func(filter + 2 * filter_col_stride, float_data_stride, vl);

  size_t overlap_num = 0;
  size_t overlap_period = 0;
  if (dilation_width_factor % stride_width == 0) {
    overlap_num = 2;
    overlap_period = dilation_width_factor / stride_width;
  } else if (2 * dilation_width_factor % stride_width == 0) {
    overlap_num = 1;
    overlap_period = 2 * dilation_width_factor / stride_width;
  }

  int32_t in_y = in_y_start;
  for (size_t out_y = 0; out_y < output_height;
       ++out_y, in_y += (int32_t)stride_height) {
    float *current_output = output + out_y * output_row_stride;

    if (in_y < 0 || in_y >= (int32_t)input_height) {
      for (size_t out_x = 0; out_x < output_width; out_x++) {
        vfloat32m4_t v_sum = init_sum_func(
            v_init_sum, vl, current_output + out_x * output_col_stride,
            load_func, float_data_stride);
        v_sum = post_process_output_func(v_sum, vl, output_activation_min,
                                         output_activation_max);
        store_func(current_output + out_x * output_col_stride,
                   float_data_stride, v_sum, vl);
      }
    } else {
      const float *current_input = input + (size_t)in_y * input_row_stride;
      if (overlap_num == 0 ||
          non_pad_out_x_start + overlap_period > non_pad_out_x_end) {
        skl_depthwise_conv2d_overlapped_0_sliding_vf32m4(
            current_input, v_filter_0, v_filter_1, v_filter_2, v_init_sum,
            current_output, input_width, output_width, pad_width, stride_width,
            dilation_width_factor, input_col_stride, output_col_stride,
            output_activation_min, output_activation_max, vl, float_data_stride,
            init_sum_func, post_process_output_func, load_func, store_func);
      } else {
        if ((int32_t)non_pad_out_x_start -
                (int32_t)overlap_num * (int32_t)overlap_period >=
            0) {
          skl_depthwise_conv2d_padding_sliding_vf32m4(
              v_init_sum, current_output, 0,
              non_pad_out_x_start - overlap_num * overlap_period,
              output_col_stride, output_activation_min, output_activation_max,
              vl, float_data_stride, init_sum_func, post_process_output_func,
              load_func, store_func);
        }

        if (overlap_num == 1) {
          skl_depthwise_conv2d_overlapped_1_sliding_vf32m4(
              current_input, v_filter_0, v_filter_1, v_filter_2, v_init_sum,
              current_output, input_width, output_width, pad_width,
              stride_width, dilation_width_factor, input_col_stride,
              output_col_stride, output_activation_min, output_activation_max,
              overlap_period, non_pad_out_x_start, non_pad_out_x_end, vl,
              float_data_stride, init_sum_func, post_process_output_func,
              load_func, store_func);
        } else {
          skl_depthwise_conv2d_overlapped_2_sliding_vf32m4(
              current_input, v_filter_0, v_filter_1, v_filter_2, v_init_sum,
              current_output, output_width, pad_width, stride_width,
              dilation_width_factor, input_col_stride, output_col_stride,
              output_activation_min, output_activation_max, overlap_period,
              non_pad_out_x_start, non_pad_out_x_end, vl, float_data_stride,
              init_sum_func, post_process_output_func, load_func, store_func);
        }

        skl_depthwise_conv2d_padding_sliding_vf32m4(
            v_init_sum, current_output,
            non_pad_out_x_end + overlap_num * overlap_period, output_width,
            output_col_stride, output_activation_min, output_activation_max, vl,
            float_data_stride, init_sum_func, post_process_output_func,
            load_func, store_func);
      }
    }
  }
}

SKL_FUNC_PRIVATE void skl_depthwise_conv2d_process_3x3_vf32m4(
    const float *input, const float *filter, vfloat32m4_t v_init_sum,
    float *output, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, size_t pad_height,
    size_t pad_width, size_t stride_height, size_t stride_width,
    size_t dilation_height_factor, size_t dilation_width_factor,
    size_t input_row_stride, size_t input_col_stride, size_t filter_row_stride,
    size_t filter_col_stride, size_t output_row_stride,
    size_t output_col_stride, float output_activation_min,
    float output_activation_max, size_t vl, size_t float_data_stride,
    vfloat32m4_t (*load_func)(const float *, size_t, size_t),
    void (*store_func)(float *, size_t, vfloat32m4_t, size_t)) {
  size_t non_pad_out_x_start =
      (pad_width > 0) ? (pad_width - 1) / stride_width + 1 : 0;
  size_t non_pad_out_x_end =
      (input_width + pad_width - 2 * dilation_width_factor > 0)
          ? (input_width + pad_width - 2 * dilation_width_factor - 1) /
                    stride_width +
                1
          : 0;

  // Slide three 1x3 filters instead of sliding one 3x3 filter to decrease
  // register pressure.

  // Slide the 0-th filter row
  // Each partial dot-product is initilized by `v_init_sum`
  // Partial dot-product will be stored to output buffer
  skl_depthwise_conv2d_process_1x3_vf32m4(
      input, filter + 0 * filter_row_stride, v_init_sum, output, input_height,
      input_width, output_height, output_width, pad_width, stride_height,
      stride_width, dilation_width_factor, input_row_stride, input_col_stride,
      filter_col_stride, output_row_stride, output_col_stride,
      output_activation_min, output_activation_max, non_pad_out_x_start,
      non_pad_out_x_end,
      -(int32_t)pad_height + 0 * (int32_t)dilation_height_factor, vl,
      float_data_stride, skl_depthwise_conv2d_init_accum_by_vector_vf32m4,
      skl_depthwise_conv2d_post_process_output_linear_vf32m4, load_func,
      store_func);

  // Slide the 1-th filter row:%s
  // Previous partial dot-product is loaded from output buffer
  // Partial dot-product will be stored to output-buffer
  skl_depthwise_conv2d_process_1x3_vf32m4(
      input, filter + 1 * filter_row_stride, v_init_sum, output, input_height,
      input_width, output_height, output_width, pad_width, stride_height,
      stride_width, dilation_width_factor, input_row_stride, input_col_stride,
      filter_col_stride, output_row_stride, output_col_stride,
      output_activation_min, output_activation_max, non_pad_out_x_start,
      non_pad_out_x_end,
      -(int32_t)pad_height + 1 * (int32_t)dilation_height_factor, vl,
      float_data_stride,
      skl_depthwise_conv2d_init_accum_by_partial_accum_vf32m4,
      skl_depthwise_conv2d_post_process_output_linear_vf32m4, load_func,
      store_func);

  // Slide the 2-th filter row
  // Previous partial dot-product is loaded from output buffer
  // Dot-product will be clamped and stored to output-buffer
  skl_depthwise_conv2d_process_1x3_vf32m4(
      input, filter + 2 * filter_row_stride, v_init_sum, output, input_height,
      input_width, output_height, output_width, pad_width, stride_height,
      stride_width, dilation_width_factor, input_row_stride, input_col_stride,
      filter_col_stride, output_row_stride, output_col_stride,
      output_activation_min, output_activation_max, non_pad_out_x_start,
      non_pad_out_x_end,
      -(int32_t)pad_height + 2 * (int32_t)dilation_height_factor, vl,
      float_data_stride,
      skl_depthwise_conv2d_init_accum_by_partial_accum_vf32m4,
      skl_depthwise_conv2d_post_process_output_linear_vf32m4, load_func,
      store_func);
}

SKL_FUNC_PRIVATE void
skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f32_f32_f32_internal_zve32f(
    float *output, const float *input, const float *filter, const float *bias,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t depth_multiplier, size_t pad_height, size_t pad_width,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    float output_activation_min, float output_activation_max) {
  (void)output_channel;
  size_t float_data_stride = depth_multiplier * sizeof(float);

  for (size_t channel = input_channel, vl = 0; channel > 0; channel -= vl) {
    vl = __riscv_vsetvl_e32m4(channel);

    if (depth_multiplier == 1) {
      // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
      vfloat32m4_t v_init_sum = __riscv_vundefined_f32m4();
      // NOLINTEND(clang-analyzer-deadcode.DeadStores)
      if (bias != NULL) {
        v_init_sum = __riscv_vle32_v_f32m4(bias, vl);
      } else {
        v_init_sum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      }

      skl_depthwise_conv2d_process_3x3_vf32m4(
          input, filter, v_init_sum, output, input_height, input_width,
          output_height, output_width, pad_height, pad_width, stride_height,
          stride_width, dilation_height_factor, dilation_width_factor,
          input_row_stride, input_col_stride, filter_row_stride,
          filter_col_stride, output_row_stride, output_col_stride,
          output_activation_min, output_activation_max, vl, float_data_stride,
          skl_depthwise_conv2d_unit_stride_load_vf32m4,
          skl_depthwise_conv2d_unit_stride_store_vf32m4);

      input += vl;
      filter += vl;
      if (bias != NULL) {
        bias += vl;
      }
      output += vl;
    } else {
      for (size_t m = 0; m < depth_multiplier; m++) {
        // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
        vfloat32m4_t v_init_sum = __riscv_vundefined_f32m4();
        // NOLINTEND(clang-analyzer-deadcode.DeadStores)
        if (bias != NULL) {
          v_init_sum =
              __riscv_vlse32_v_f32m4(bias, (ptrdiff_t)float_data_stride, vl);
        } else {
          v_init_sum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }

        skl_depthwise_conv2d_process_3x3_vf32m4(
            input, filter, v_init_sum, output, input_height, input_width,
            output_height, output_width, pad_height, pad_width, stride_height,
            stride_width, dilation_height_factor, dilation_width_factor,
            input_row_stride, input_col_stride, filter_row_stride,
            filter_col_stride, output_row_stride, output_col_stride,
            output_activation_min, output_activation_max, vl, float_data_stride,
            skl_depthwise_conv2d_strided_load_vf32m4,
            skl_depthwise_conv2d_strided_store_vf32m4);

        filter++;
        if (bias != NULL) {
          bias++;
        }
        output++;
      }

      input += vl;
      filter += (vl - 1) * depth_multiplier;
      if (bias != NULL) {
        bias += (vl - 1) * depth_multiplier;
      }
      output += (vl - 1) * depth_multiplier;
    }
  }
}

SKL_FUNC void skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f32_f32_f32_zve32f(
    float *output, const float *input, const float *filter, size_t input_height,
    size_t input_width, size_t input_channel, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride) {
  skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f32_f32_f32_internal_zve32f(
      output, input, filter, NULL, input_height, input_width, input_channel,
      output_height, output_width, output_channel, depth_multiplier, 0, 0,
      stride_height, stride_width, dilation_height_factor,
      dilation_width_factor, input_row_stride, input_col_stride,
      filter_row_stride, filter_col_stride, output_row_stride,
      output_col_stride, 0.0f, 0.0f);
}
