// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

// IWYU pragma: begin_exports

/*
 * GEMM Kernels
 */

#if defined(__riscv_zve32x)
#include "gemm/rvv/gemm_i8_i8_i32_zve32x.h"
#include "gemm/rvv/gemm_i8_i8_i32_zve32x_x390.h"
#endif

#if defined(__riscv_zvfh)
#include "gemm/rvv/gemm_f16_f16_f16_zvfh_x390.h"
#include "gemm/rvv/gemm_f16_f16_f32_zvfh.h"
#include "gemm/rvv/gemm_f16_f16_f32_zvfh_x390.h"
#endif

#if defined(__riscv_zvfbfwma)
#include "gemm/rvv/gemm_bf16_bf16_f32_zvfbfwma.h"
#include "gemm/rvv/gemm_bf16_bf16_f32_zvfbfwma_x390.h"
#endif

#if defined(__riscv_zve32f)
#include "gemm/rvv/gemm_f32_f32_f32_zve32f.h"
#include "gemm/rvv/gemm_f32_f32_f32_zve32f_x390.h"
#endif

#if defined(__riscv_zve64d)
#include "gemm/rvv/gemm_f64_f64_f64_zve64d.h"
#include "gemm/rvv/gemm_f64_f64_f64_zve64d_x390.h"
#endif

#if defined(__riscv_xsfmm32a8f)
#include "gemm/xsfmm/gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f.h"
#include "gemm/xsfmm/gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f.h"
#endif

#if defined(__riscv_xsfmm32a8i)
#include "gemm/xsfmm/gemm_a1b01_i8c_i8_i32_xsfmm32a8i.h"
#endif

#if defined(__riscv_xsfmm32a16f)
#include "gemm/xsfmm/gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f.h"
#include "gemm/xsfmm/gemm_a1b01_f16c_f16_f32_xsfmm32a16f.h"
#endif

#if defined(__riscv_xsfmm32a32f)
#include "gemm/xsfmm/gemm_f32c_f32_f32_xsfmm32a32f.h"
#endif

#if defined(__riscv_xsfvqdotq)
#include "gemm/xsfvqdotq/gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq.h"
#endif

/*
 * Transpose Kernels
 */

#if defined(__riscv_xsfmmbase)
#include "transpose/xsfmm/transpose_e16_xsfmmbase.h"
#include "transpose/xsfmm/transpose_e32_xsfmmbase.h"
#include "transpose/xsfmm/transpose_e8_xsfmmbase.h"
#endif

/*
 * Exponential Function Kernels
 */

#if defined(__riscv_zvfh)
#include "exp/exp_f16_zvfh.h"
#endif
#if defined(__riscv_xsfvfexp16e)
#include "exp/exp_f16_xsfvfexp16e.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh)
#include "exp/exp_f16_xsfvfexpa_zvfh.h"
#endif

#if defined(__riscv_zve32f)
#include "exp/exp_bf16_zve32f.h"
#endif
#if defined(__riscv_xsfvfbfa)
#include "exp/exp_bf16_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e)
#include "exp/exp_bf16_xsfvfbfexp16e.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin)
#include "exp/exp_bf16_xsfvfexpa_zvfbfmin.h"
#endif
#if defined(__riscv_zvfbfmin)
#include "exp/exp_bf16_zvfbfmin.h"
#endif

#if defined(__riscv_zve32f)
#include "exp/exp_f32_zve32f.h"
#endif
#if defined(__riscv_xsfvfexp32e)
#include "exp/exp_f32_xsfvfexp32e.h"
#endif
#if defined(__riscv_xsfvfexpa)
#include "exp/exp_f32_xsfvfexpa.h"
#endif

/*
 * Sigmoid Function Kernels
 */

#if defined(__riscv_xsfvfbfa)
#include "sigmoid/sigmoid_bf16_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_xsfvfbfa)
#include "sigmoid/sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfexp32e)
#include "sigmoid/sigmoid_bf16_xsfvfexp32e.h"
#endif
#if defined(__riscv_xsfvfexpa)
#include "sigmoid/sigmoid_bf16_xsfvfexpa.h"
#endif
#if defined(__riscv_zve32f)
#include "sigmoid/sigmoid_bf16_zve32f.h"
#endif

#if defined(__riscv_zvfh)
#include "sigmoid/sigmoid_f16_zvfh.h"
#endif
#if defined(__riscv_xsfvfexp16e)
#include "sigmoid/sigmoid_f16_xsfvfexp16e.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh)
#include "sigmoid/sigmoid_f16_xsfvfexpa_zvfh.h"
#endif

#if defined(__riscv_zve32f)
#include "sigmoid/sigmoid_f32_zve32f.h"
#endif
#if defined(__riscv_xsfvfexp32e)
#include "sigmoid/sigmoid_f32_xsfvfexp32e.h"
#endif
#if defined(__riscv_xsfvfexpa)
#include "sigmoid/sigmoid_f32_xsfvfexpa.h"
#endif

/*
 * Conversion Functions
 */
#if defined(__riscv_zvfofp8min)
#include "cvt/cvt_f32_f8_zvfofp8min.h"
#include "cvt/cvt_f8_bf16_zvfofp8min.h"
#endif

#if defined(__riscv_zvfofp8min) && defined(__riscv_zvfbfmin)
#include "cvt/cvt_bf16_f8_zvfofp8min_zvfbfmin.h"
#endif

#if defined(__riscv_zvfofp4min)
#include "cvt/cvt_f4_f8_zvfofp4min.h"
#endif

/*
 * Depthwise Convolution Kernels
 */

#if defined(__riscv_zve32f)
#include "depthwise_conv2d/depthwise_conv2d_f32_f32_f32_zve32f.h"
#endif

#if defined(__riscv_zve32x)
#include "depthwise_conv2d/depthwise_conv2d_i8_i8_i32_zve32x.h"
#endif

#if defined(__riscv_zvfh)
#include "depthwise_conv2d/depthwise_conv2d_f16_f16_f16_zvfh.h"
#endif

/*
 * Softmax Kernels
 */

#if defined(__riscv_zvfh)
#include "softmax/softmax_f16_zvfh.h"
#endif
#if defined(__riscv_xsfvfexp16e)
#include "softmax/softmax_f16_xsfvfexp16e.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh)
#include "softmax/softmax_f16_xsfvfexpa_zvfh.h"
#endif

#if defined(__riscv_zve32f)
#include "softmax/softmax_bf16_zve32f.h"
#endif
#if defined(__riscv_xsfvfbfa)
#include "softmax/softmax_bf16_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_xsfvfbfa)
#include "softmax/softmax_bf16_xsfvfbfexp16e_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_zvfbfmin)
#include "softmax/softmax_bf16_xsfvfbfexp16e_zvfbfmin.h"
#endif
#if defined(__riscv_xsfvfexp32e) && defined(__riscv_zvfbfmin)
#include "softmax/softmax_bf16_xsfvfexp32e_zvfbfmin.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_xsfvfbfa)
#include "softmax/softmax_bf16_xsfvfexpa_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin)
#include "softmax/softmax_bf16_xsfvfexpa_zvfbfmin.h"
#endif
#if defined(__riscv_zvfbfmin)
#include "softmax/softmax_bf16_zvfbfmin.h"
#endif

#if defined(__riscv_zve32f)
#include "softmax/softmax_f32_zve32f.h"
#endif
#if defined(__riscv_xsfvfexp32e)
#include "softmax/softmax_f32_xsfvfexp32e.h"
#endif
#if defined(__riscv_xsfvfexpa)
#include "softmax/softmax_f32_xsfvfexpa.h"
#endif

/*
 * GELU Kernels
 */

#if defined(__riscv_zve32f)
#include "gelu/gelu_f32_zve32f.h"
#endif

/*
 * Pack Kernels
 */
#if defined(__riscv_zve32x)
#include "pack/rvv/skl_pack_e16_zve32x.h"
#include "pack/rvv/skl_pack_e32_zve32x.h"
#include "pack/rvv/skl_pack_e8_zve32x.h"
#endif

// IWYU pragma: end_exports
