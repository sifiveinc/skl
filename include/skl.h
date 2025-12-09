// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// IWYU pragma: begin_exports

/*
 * GEMM Kernels
 */

#include "../src/gemm/scalar/gemm_bf16rc_bf16rc_f32rc_scalar.h"
#include "../src/gemm/scalar/gemm_f16rc_f16rc_f16rc_scalar.h"
#include "../src/gemm/scalar/gemm_f16rc_f16rc_f32rc_scalar.h"
#include "../src/gemm/scalar/gemm_f32rc_f32rc_f32rc_scalar.h"
#include "../src/gemm/scalar/gemm_f64rc_f64rc_f64rc_scalar.h"
#include "../src/gemm/scalar/gemm_f8e4m3rc_f8e4m3rc_f32rc_scalar.h"
#include "../src/gemm/scalar/gemm_i8rc_i8rc_i32rc_scalar.h"

#include "../src/gemm/scalar/gemm_bf16rcprc_bf16rcprc_f32rcprc_scalar.h"
#include "../src/gemm/scalar/gemm_f16rcprc_f16rcprc_f32rcprc_scalar.h"
#include "../src/gemm/scalar/gemm_f32rcprc_f32rcprc_f32rcprc_scalar.h"
#include "../src/gemm/scalar/gemm_f64rcprc_f64rcprc_f64rcprc_scalar.h"
#include "../src/gemm/scalar/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc_scalar.h"
#include "../src/gemm/scalar/gemm_i8rcprc_i8rcprc_i32rcprc_scalar.h"

#if defined(__riscv_zve32x)
#include "../src/gemm/rvv/gemm_i8_i8_i32_zve32x_x390.h"
#endif

#if defined(__riscv_zvfh)
#include "../src/gemm/rvv/gemm_f16_f16_f16_zvfh_x390.h"
#include "../src/gemm/rvv/gemm_f16_f16_f32_zvfh_x390.h"
#endif

#if defined(__riscv_zve32f)
#include "../src/gemm/rvv/gemm_f32_f32_f32_zve32f_x390.h"
#endif

#if defined(__riscv_zve64d)
#include "../src/gemm/rvv/gemm_f64_f64_f64_zve64d_x390.h"
#endif

#if defined(__riscv_xsfmm32a8f)
#include "../src/gemm/xsfmm/gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f.h"
#endif

#if defined(__riscv_xsfmm32a8i)
#include "../src/gemm/xsfmm/gemm_a1b01_i8c_i8_i32_xsfmm32a8i.h"
#endif

#if defined(__riscv_xsfmm32a16f)
#include "../src/gemm/xsfmm/gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f.h"
#include "../src/gemm/xsfmm/gemm_a1b01_f16c_f16_f32_xsfmm32a16f.h"
#endif

#if defined(__riscv_xsfmm32a32f)
#include "../src/gemm/xsfmm/gemm_a1b01_f32c_f32_f32_xsfmm32a32f.h"
#endif

#if defined(__riscv_xsfvqdotq)
#include "../src/gemm/xsfvqdotq/gemm_a1b01_i8_i8pc_i32_xsfvqdotq.h"
#endif

/*
 * Matrix packing Kernels
 */

#if defined(__riscv_zve32x)
#include "../src/gemm/xsfvqdotq/pack_b_xsfvqdotq.h"
#endif

/*
 * Transpose Kernels
 */

#include "../src/transpose/scalar/transpose_e16_scalar.h"
#include "../src/transpose/scalar/transpose_e32_scalar.h"
#include "../src/transpose/scalar/transpose_e8_scalar.h"

#if defined(__riscv_zve32x)
#include "../src/transpose/rvv/transpose_e16_zve32x.h"
#include "../src/transpose/rvv/transpose_e32_zve32x.h"
#include "../src/transpose/rvv/transpose_e8_zve32x.h"
#endif

#if defined(__riscv_xsfmmbase)
#include "../src/transpose/xsfmm/transpose_e16_xsfmmbase.h"
#include "../src/transpose/xsfmm/transpose_e32_xsfmmbase.h"
#include "../src/transpose/xsfmm/transpose_e8_xsfmmbase.h"
#endif

/*
 * Exponential and Logistic Function Kernels
 */

#include "../src/logistic/logistic_f16_scalar.h"
#include "../src/logistic/logistic_f32_scalar.h"

#if defined(__riscv_zvfbfmin)
#include "../src/exp/exp_bf16_zvfbfmin.h"
#endif

#if defined(__riscv_xsfvfbfa)
#include "../src/exp/exp_bf16_xsfvfbfa.h"
#endif

#if defined(__riscv_zvfh)
#include "../src/exp/exp_f16_zvfh.h"
#include "../src/logistic/logistic_f16_zvfh.h"
#endif

#if defined(__riscv_zve32f)
#include "../src/exp/exp_bf16_zve32f.h"
#include "../src/exp/exp_f32_zve32f.h"
#include "../src/logistic/logistic_f32_zve32f.h"
#endif

#if defined(__riscv_xsfvfbfexp16e)
#include "../src/exp/exp_bf16_xsfvfbfexp16e.h"
#endif

#if defined(__riscv_xsfvfexp16e)
#include "../src/exp/exp_f16_xsfvfexp16e.h"
#include "../src/logistic/logistic_f16_xsfvfexp16e.h"
#endif

#if defined(__riscv_xsfvfexp32e)
#include "../src/exp/exp_f32_xsfvfexp32e.h"
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin)
#include "../src/exp/exp_bf16_xsfvfexpa_zvfbfmin.h"
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh)
#include "../src/exp/exp_f16_xsfvfexpa_zvfh.h"
#endif

#if defined(__riscv_xsfvfexpa)
#include "../src/exp/exp_f32_xsfvfexpa.h"
#include "../src/logistic/logistic_f32_xsfvfexpa.h"
#endif

/*
 * Conversion Functions
 */
#if defined(__riscv_zvfofp8min)
#include "../src/cvt/cvt_zvfofp8min.h"
#endif

#if defined(__riscv_zvfofp8min) && defined(__riscv_zvfbfmin)
#include "../src/cvt/cvt_zvfofp8min_zvfbfmin.h"
#endif

#if defined(__riscv_zvfofp4min)
#include "../src/cvt/cvt_zvfofp4min.h"
#endif

/*
 * Depthwise Convolution Kernels
 */

#if defined(__riscv_zfh)
#include "../src/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_scalar.h"
#endif

#include "../src/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_scalar.h"
#include "../src/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_scalar.h"

#if defined(__riscv_zve32f)
#include "../src/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_zve32f.h"
#endif

#if defined(__riscv_zve32x)
#include "../src/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_zve32x.h"
#endif

#if defined(__riscv_zvfh)
#include "../src/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_zvfh.h"
#endif

/*
 * Softmax Kernels
 */

#include "../src/softmax/softmax_f16_scalar.h"
#if defined(__riscv_zvfh)
#include "../src/softmax/softmax_f16_zvfh.h"
#endif
#if defined(__riscv_xsfvfexp16e)
#include "../src/softmax/softmax_f16_xsfvfexp16e.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh)
#include "../src/softmax/softmax_f16_xsfvfexpa_zvfh.h"
#endif

#include "../src/softmax/softmax_bf16_scalar.h"
#if defined(__riscv_zve32f)
#include "../src/softmax/softmax_bf16_zve32f.h"
#endif
#if defined(__riscv_xsfvfbfa)
#include "../src/softmax/softmax_bf16_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_xsfvfbfa)
#include "../src/softmax/softmax_bf16_xsfvfbfexp16e_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_zvfbfmin)
#include "../src/softmax/softmax_bf16_xsfvfbfexp16e_zvfbfmin.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_xsfvfbfa)
#include "../src/softmax/softmax_bf16_xsfvfexpa_xsfvfbfa.h"
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin)
#include "../src/softmax/softmax_bf16_xsfvfexpa_zvfbfmin.h"
#endif
#if defined(__riscv_zvfbfmin)
#include "../src/softmax/softmax_bf16_zvfbfmin.h"
#endif

#include "../src/softmax/softmax_f32_scalar.h"
#if defined(__riscv_zve32f)
#include "../src/softmax/softmax_f32_zve32f.h"
#endif
#if defined(__riscv_xsfvfexp32e)
#include "../src/softmax/softmax_f32_xsfvfexp32e.h"
#endif
#if defined(__riscv_xsfvfexpa)
#include "../src/softmax/softmax_f32_xsfvfexpa.h"
#endif

/*
 * GELU Kernels
 */

#if defined(__riscv_zve32f)
#include "../src/gelu/gelu_f32_zve32f.h"
#endif

// IWYU pragma: end_exports
