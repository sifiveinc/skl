// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_zve32x(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point) {
  (void)input_height;
  (void)input_width;

  int8_t input_zero_point_i8 = (int8_t)input_zero_point;

  size_t input_next_col_offset = stride_width * input_col_stride;
  size_t input_next_row_remaining_offset =
      stride_height * input_row_stride - output_width * input_next_col_offset;

  size_t output_next_row_remaining_offset =
      output_row_stride - output_width * output_col_stride;

  if (depth_multiplier == 1) {
    for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
         output_c_avl -= vl) {
      vl = __riscv_vsetvl_e8m2(output_c_avl);

      const int8_t *input_ptr = input;
      int32_t *output_ptr = output;
      for (size_t output_h = 0; output_h < output_height; output_h++) {
        for (size_t output_w = 0; output_w < output_width; output_w++) {
          vint32m8_t v_output_i32 = __riscv_vmv_v_x_i32m8(0, vl);

          for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
            for (size_t filter_w = 0; filter_w < filter_width; filter_w++) {
              vint8m2_t v_input_i8 = __riscv_vle8_v_i8m2(
                  input_ptr +
                      filter_h * dilation_height_factor * input_row_stride +
                      filter_w * dilation_width_factor * input_col_stride,
                  vl);
              vint8m2_t v_filter_i8 =
                  __riscv_vle8_v_i8m2(filter + filter_h * filter_row_stride +
                                          filter_w * filter_col_stride,
                                      vl);

              vint16m4_t v_input_i16 =
                  __riscv_vwsub_vx_i16m4(v_input_i8, input_zero_point_i8, vl);
              vint16m4_t v_filter_i16 =
                  __riscv_vwadd_vx_i16m4(v_filter_i8, 0, vl);
              v_output_i32 = __riscv_vwmacc_vv_i32m8(v_output_i32, v_input_i16,
                                                     v_filter_i16, vl);
            }
          }

          __riscv_vse32_v_i32m8(output_ptr, v_output_i32, vl);

          input_ptr += input_next_col_offset;
          output_ptr += output_col_stride;
        }

        input_ptr += input_next_row_remaining_offset;
        output_ptr += output_next_row_remaining_offset;
      }
      input += vl;
      filter += vl;
      output += vl;
    }
  } else {
    if (input_channel == 1) {
      for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
           output_c_avl -= vl) {
        vl = __riscv_vsetvl_e8m2(output_c_avl);

        const int8_t *input_ptr = input;
        int32_t *output_ptr = output;
        for (size_t output_h = 0; output_h < output_height; output_h++) {
          for (size_t output_w = 0; output_w < output_width; output_w++) {
            vint32m8_t v_output_i32 = __riscv_vmv_v_x_i32m8(0, vl);

            for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
              for (size_t filter_w = 0; filter_w < filter_width; filter_w++) {
                vint8m2_t v_input_i8 = __riscv_vlse8_v_i8m2(
                    input_ptr +
                        filter_h * dilation_height_factor * input_row_stride +
                        filter_w * dilation_width_factor * input_col_stride,
                    0, vl);
                vint8m2_t v_filter_i8 =
                    __riscv_vle8_v_i8m2(filter + filter_h * filter_row_stride +
                                            filter_w * filter_col_stride,
                                        vl);

                vint16m4_t v_input_i16 =
                    __riscv_vwsub_vx_i16m4(v_input_i8, input_zero_point_i8, vl);
                vint16m4_t v_filter_i16 =
                    __riscv_vwadd_vx_i16m4(v_filter_i8, 0, vl);
                v_output_i32 = __riscv_vwmacc_vv_i32m8(
                    v_output_i32, v_input_i16, v_filter_i16, vl);
              }
            }

            __riscv_vse32_v_i32m8(output_ptr, v_output_i32, vl);

            input_ptr += input_next_col_offset;
            output_ptr += output_col_stride;
          }

          input_ptr += input_next_row_remaining_offset;
          output_ptr += output_next_row_remaining_offset;
        }
        filter += vl;
        output += vl;
      }
    } else {
      size_t filter_stride = depth_multiplier * sizeof(int8_t);
      size_t output_stride = depth_multiplier * sizeof(int32_t);

      for (size_t input_c_avl = input_channel, vl; input_c_avl > 0;
           input_c_avl -= vl) {
        vl = __riscv_vsetvl_e8m2(input_c_avl);

        for (size_t m = 0; m < depth_multiplier; m++) {
          const int8_t *input_ptr = input;
          int32_t *output_ptr = output;
          for (size_t output_h = 0; output_h < output_height; output_h++) {
            for (size_t output_w = 0; output_w < output_width; output_w++) {
              vint32m8_t v_output_i32 = __riscv_vmv_v_x_i32m8(0, vl);

              for (size_t filter_h = 0; filter_h < filter_height; filter_h++) {
                for (size_t filter_w = 0; filter_w < filter_width; filter_w++) {
                  vint8m2_t v_input_i8 = __riscv_vle8_v_i8m2(
                      input_ptr +
                          filter_h * dilation_height_factor * input_row_stride +
                          filter_w * dilation_width_factor * input_col_stride,
                      vl);
                  vint8m2_t v_filter_i8 = __riscv_vlse8_v_i8m2(
                      filter + filter_h * filter_row_stride +
                          filter_w * filter_col_stride,
                      (ptrdiff_t)filter_stride, vl);

                  vint16m4_t v_input_i16 = __riscv_vwsub_vx_i16m4(
                      v_input_i8, input_zero_point_i8, vl);
                  vint16m4_t v_filter_i16 =
                      __riscv_vwadd_vx_i16m4(v_filter_i8, 0, vl);
                  v_output_i32 = __riscv_vwmacc_vv_i32m8(
                      v_output_i32, v_input_i16, v_filter_i16, vl);
                }
              }

              __riscv_vsse32_v_i32m8(output_ptr, (ptrdiff_t)output_stride,
                                     v_output_i32, vl);

              input_ptr += input_next_col_offset;
              output_ptr += output_col_stride;
            }

            input_ptr += input_next_row_remaining_offset;
            output_ptr += output_next_row_remaining_offset;
          }

          filter++;
          output++;
        }

        input += vl;
        filter += (vl - 1) * depth_multiplier;
        output += (vl - 1) * depth_multiplier;
      }
    }
  }
}

/* clang-format off */
/*
  Strategy for allocating vector registers:
  v0-v7: accumulation for multiplication of each input and the corresponding filter
  v8-v11: dynamic input data / product of input and filter / dynamic filter data 
  v12-v15: dynamic input data / product of input and filter data
  v16-v19: dynamic input data / product of input and filter data
  v20-v21: the 0-th filter data
  v22-v23: the 1-th filter data
  v24-v25: the 2-th filter data
  v26-v27: the 3-th filter data
  v28-v29: the 5-th filter data
  v30-v31: the 7-th filter data

  Keeping as many filter as possible in vector registers to decrease loading pressure.
*/
/* clang-format on */

// Pre-load data out of the main loop from `F3x3DotProductI8I32`
SKL_FUNC_PRIVATE void skl_depthwise_conv2d_f3x3_dot_product_i8_i32_pre_load(
    const int8_t *input[3][3], const int8_t *filter[3][3], size_t vl) {
  __asm__ volatile("vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v20, (%[filter00])\n\t"
                   "vle8.v v22, (%[filter01])\n\t"
                   "vle8.v v24, (%[filter02])\n\t"
                   "vle8.v v26, (%[filter10])\n\t"
                   "vle8.v v28, (%[filter12])\n\t"
                   "vle8.v v30, (%[filter21])\n\t"
                   "vle8.v v10, (%[input00])\n\t"
                   "vle8.v v14, (%[input01])\n\t"
                   "vle8.v v18, (%[input02])\n\t"
                   :
                   : [vl] "r"(vl), [filter00] "r"(filter[0][0]),
                     [filter01] "r"(filter[0][1]), [filter02] "r"(filter[0][2]),
                     [filter10] "r"(filter[1][0]), [filter12] "r"(filter[1][2]),
                     [filter21] "r"(filter[2][1]), [input00] "r"(input[0][0]),
                     [input01] "r"(input[0][1]), [input02] "r"(input[0][2])
                   : "v10", "v11", "v14", "v15", "v18", "v19", "v20", "v21",
                     "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                     "v30", "v31", "vtype", "vl");
}

// Dot product `vl` elements for input and 3x3 filter
SKL_FUNC_PRIVATE void skl_depthwise_conv2d_f3x3_dot_product_i8_i32(
    const int8_t *input[3][3], const int8_t *filter[3][3],
    int32_t *output, // NOLINT(readability-non-const-parameter)
    size_t vl,
    size_t (*post_process_input_func)(const int8_t *[3][3], size_t,
                                      const void *),
    const void *post_process_input_func_args) {
  /* clang-format off */
  /* Current vector register status (the number of vector register for each below field is 2)
  | v0  | v2  | v4  | v6  | v8  | v10 | v12 | v14 | v16 | v18 | v20 | v22 | v24 | v26 | v28 | v30 |
  |  x  |  x  |  x  |  x  |  x  | in0 |  x  | in1 |  x  | in2 | f0  | f1  | f2  | f3  | f5  | f7  |
  
  "x" for don't care
  
  */
  /* clang-format on */

  __asm__ volatile("vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vwmul.vv v8, v10, v20\n\t"
                   "vwmul.vv v12, v14, v22\n\t"
                   "vwmul.vv v16, v18, v24\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.vv v0, v8, v12\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v8, (%[filter11])\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v16\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v14, (%[input10])\n\t"
                   "vle8.v v18, (%[input11])\n\t"
                   :
                   : [vl] "r"(vl), [input10] "r"(input[1][0]),
                     [input11] "r"(input[1][1]), [filter11] "r"(filter[1][1])
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                     "v18", "v19", "vtype", "vl");
  /* clang-format off */
  /* Current vector register status after the above instructions
  | v0  | v2  | v4  | v6  | v8  | v10 | v12 | v14 | v16 | v18 | v20 | v22 | v24 | v26 | v28 | v30 |
  | sum(0+1+2)            | f4  |  x  |  x  | in3 |  x  | in4 | f0  | f1  | f2  | f3  | f5  | f7  |
  */
  /* clang-format on */

  __asm__ volatile("vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vwmul.vv v12, v14, v26\n\t"
                   "vwmul.vv v16, v18, v8\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v12\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v8, (%[filter20])\n\t"
                   "vle8.v v14, (%[input12])\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v16\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v18, (%[input20])\n\t"
                   :
                   : [vl] "r"(vl), [input12] "r"(input[1][2]),
                     [input20] "r"(input[2][0]), [filter20] "r"(filter[2][0])
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                     "vtype", "vl");
  /* clang-format off */
  /* Current vector register status after the above instructions
  | v0  | v2  | v4  | v6  | v8  | v10 | v12 | v14 | v16 | v18 | v20 | v22 | v24 | v26 | v28 | v30 |
  | sum(0+1+2+3+4)        | f6  |  x  |  x  | in5 |  x  | in6 | f0  | f1  | f2  | f3  | f5  | f7  |
  */
  /* clang-format on */

  __asm__ volatile("vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vwmul.vv v12, v14, v28\n\t"
                   "vwmul.vv v16, v18, v8\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v12\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v8, (%[filter22])\n\t"
                   "vle8.v v14, (%[input21])\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v16\n\t"
                   "vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vle8.v v18, (%[input22])\n\t"
                   :
                   : [vl] "r"(vl), [input21] "r"(input[2][1]),
                     [input22] "r"(input[2][2]), [filter22] "r"(filter[2][2])
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                     "vtype", "vl");
  /* clang-format off */
  /* Current vector register status after the above instructions
  | v0  | v2  | v4  | v6  | v8  | v10 | v12 | v14 | v16 | v18 | v20 | v22 | v24 | v26 | v28 | v30 |
  | sum(0+1+2+3+4+5+6)    | f8  |  x  |  x  | in7 |  x  | in8 | f0  | f1  | f2  | f3  | f5  | f7  |
  */
  /* clang-format on */

  __asm__ volatile("vsetvli zero, %[vl], e8, m2, ta, ma\n\t"
                   "vwmul.vv v12, v14, v30\n\t"
                   "vwmul.vv v16, v18, v8\n\t"
                   "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
                   "vwadd.wv v0, v0, v12\n\t"
                   :
                   : [vl] "r"(vl)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v12",
                     "v13", "v14", "v15", "v16", "v17", "v18", "v19", "vtype",
                     "vl");
  size_t next_vl =
      post_process_input_func(input, vl, post_process_input_func_args);
  __asm__ volatile(
      "vsetvli zero, %[next_vl], e8, m2, ta, ma\n\t"
      "vle8.v v10, (%[next_input00])\n\t"
      "vle8.v v14, (%[next_input01])\n\t"
      "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
      "vwadd.wv v0, v0, v16\n\t"
      "vsetvli zero, %[next_vl], e8, m2, ta, ma\n\t"
      "vle8.v v18, (%[next_input02])\n\t"
      :
      : [vl] "r"(vl), [next_vl] "r"(next_vl), [next_input00] "r"(input[0][0]),
        [next_input01] "r"(input[0][1]), [next_input02] "r"(input[0][2])
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "vtype", "vl");
  /* clang-format off */
  /* Current vector register status after the above instructions
  | v0  | v2  | v4  | v6  | v8  | v10 | v12 | v14 | v16 | v18 | v20 | v22 | v24 | v26 | v28 | v30 |
  | sum(0+1+2+3+4+5+6+7+8)|  x  | in0 |  x  | in1'|  x  | in2'| f0  | f1  | f2  | f3  | f5  | f7  |
                                (next)      (next)      (next)
  */
  /* clang-format on */

  __asm__ volatile("vsetvli zero, %[vl], e32, m8, ta, ma\n\t"
                   "vse32.v v0, (%[output])\n\t"
                   : [output] "+r"(output)
                   : [vl] "r"(vl)
                   : "vtype", "vl", "memory");
}

typedef struct {
  size_t *output_h, *output_w;
  size_t output_height, output_width;
  size_t input_next_col_offset;
  size_t input_next_row_remaining_offset;
} VcF3x3PostProcessArgs;

SKL_FUNC_PRIVATE size_t skl_depthwise_conv2d_vc_f3x3_post_processing_input_i8(
    const int8_t *kernel_input_ptrs[3][3], size_t vl, const void *args) {
  const VcF3x3PostProcessArgs *decoded_args =
      (const VcF3x3PostProcessArgs *)args;
  size_t output_h = *decoded_args->output_h;
  size_t output_w = *decoded_args->output_w;
  size_t output_height = decoded_args->output_height;
  size_t output_width = decoded_args->output_width;
  size_t input_next_col_offset = decoded_args->input_next_col_offset;
  size_t input_next_row_remaining_offset =
      decoded_args->input_next_row_remaining_offset;

  if (output_h == output_height - 1 && output_w == output_width - 1) {
    return 0;
  }

  if (output_w + 1 < output_width) {
#pragma unroll 3
    for (size_t filter_i = 0; filter_i < 3; filter_i++) {
#pragma unroll 3
      for (size_t filter_j = 0; filter_j < 3; filter_j++) {
        kernel_input_ptrs[filter_i][filter_j] += input_next_col_offset;
      }
    }
  } else {
#pragma unroll 3
    for (size_t filter_i = 0; filter_i < 3; filter_i++) {
#pragma unroll 3
      for (size_t filter_j = 0; filter_j < 3; filter_j++) {
        kernel_input_ptrs[filter_i][filter_j] +=
            input_next_row_remaining_offset;
      }
    }
  }
  return vl;
}

// Make `output` subtract sum {z_in * f_i}
SKL_FUNC_PRIVATE void skl_depthwise_conv2d_vc_f3x3_apply_input_zp(
    const int8_t *filter, int32_t *output, size_t output_height,
    size_t output_width, size_t output_channel, size_t filter_row_stride,
    size_t filter_col_stride, size_t output_row_stride,
    size_t output_col_stride, int32_t input_zero_point) {
  size_t output_next_row_remaining_offset =
      output_row_stride - output_width * output_col_stride;

  int8_t input_zero_point_i8 = (int8_t)input_zero_point;

  for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
       output_c_avl -= vl) {
    vl = __riscv_vsetvl_e8m2(output_c_avl);

    vint32m8_t v_input_zp_mul_filter_sum_i32 = __riscv_vmv_v_x_i32m8(0, vl);
#pragma unroll 3
    for (size_t filter_i = 0; filter_i < 3; filter_i++) {
#pragma unroll 3
      for (size_t filter_j = 0; filter_j < 3; filter_j++) {
        vint8m2_t v_filter_i8 =
            __riscv_vle8_v_i8m2(filter + filter_j * filter_col_stride +
                                    filter_i * filter_row_stride,
                                vl);
        vint16m4_t v_input_zp_mul_filter_i16 =
            __riscv_vwmul_vx_i16m4(v_filter_i8, input_zero_point_i8, vl);
        v_input_zp_mul_filter_sum_i32 = __riscv_vwadd_wv_i32m8(
            v_input_zp_mul_filter_sum_i32, v_input_zp_mul_filter_i16, vl);
      }
    }

    int32_t *output_ptr = output;
    for (size_t output_h = 0; output_h < output_height; output_h++) {
      for (size_t output_w = 0; output_w < output_width; output_w++) {
        vint32m8_t v_output_i32 = __riscv_vle32_v_i32m8(output_ptr, vl);
        v_output_i32 = __riscv_vsub_vv_i32m8(v_output_i32,
                                             v_input_zp_mul_filter_sum_i32, vl);
        __riscv_vse32_v_i32m8(output_ptr, v_output_i32, vl);

        output_ptr += output_col_stride;
      }
      output_ptr += output_next_row_remaining_offset;
    }

    filter += vl;
    output += vl;
  }
}

SKL_FUNC void skl_depthwise_conv2d_f3x3m1_i8hwc_i8hwim_i32hwc_zve32x(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point) {
  (void)input_height;
  (void)input_width;
  (void)input_channel;

  size_t input_next_col_offset = stride_width * input_col_stride;
  size_t input_next_row_remaining_offset =
      stride_height * input_row_stride -
      (output_width - 1) * input_next_col_offset;

  size_t output_next_row_remaining_offset =
      output_row_stride - output_width * output_col_stride;

  size_t e8m2_vlmax = __riscv_v_min_vlen >> 2; // VLEN * 2 / 8

  for (size_t output_c_avl = output_channel, vl; output_c_avl > 0;
       output_c_avl -= vl) {
    vl = (output_c_avl < e8m2_vlmax) ? output_c_avl : e8m2_vlmax;

    const int8_t *kernel_input_ptrs[3][3];
    const int8_t *kernel_filter_ptrs[3][3];

#pragma unroll 3
    for (size_t i = 0; i < 3; i++) {
#pragma unroll 3
      for (size_t j = 0; j < 3; j++) {
        kernel_input_ptrs[i][j] = input +
                                  j * dilation_width_factor * input_col_stride +
                                  i * dilation_height_factor * input_row_stride;
        kernel_filter_ptrs[i][j] =
            filter + j * filter_col_stride + i * filter_row_stride;
      }
    }

    int32_t *output_ptr = output;
    size_t output_h;
    size_t output_w;

    VcF3x3PostProcessArgs vc_f3x3_post_process_args;
    vc_f3x3_post_process_args.output_h = &output_h;
    vc_f3x3_post_process_args.output_w = &output_w;
    vc_f3x3_post_process_args.output_height = output_height;
    vc_f3x3_post_process_args.output_width = output_width;
    vc_f3x3_post_process_args.input_next_col_offset = input_next_col_offset;
    vc_f3x3_post_process_args.input_next_row_remaining_offset =
        input_next_row_remaining_offset;

    skl_depthwise_conv2d_f3x3_dot_product_i8_i32_pre_load(
        kernel_input_ptrs, kernel_filter_ptrs, vl);

    for (output_h = 0; output_h < output_height; output_h++) {
      for (output_w = 0; output_w < output_width; output_w++) {
        skl_depthwise_conv2d_f3x3_dot_product_i8_i32(
            kernel_input_ptrs, kernel_filter_ptrs, output_ptr, vl,
            skl_depthwise_conv2d_vc_f3x3_post_processing_input_i8,
            (const void *)(&vc_f3x3_post_process_args));
        output_ptr += output_col_stride;
      }
      output_ptr += output_next_row_remaining_offset;
    }

    input += vl;
    filter += vl;
    output += vl;
  }

  // `F3x3DotProductI8I32` doesn't include calculation about input_zero_point
  output -= output_channel;
  filter -= output_channel;
  skl_depthwise_conv2d_vc_f3x3_apply_input_zp(
      filter, output, output_height, output_width, output_channel,
      filter_row_stride, filter_col_stride, output_row_stride,
      output_col_stride, input_zero_point);
}
