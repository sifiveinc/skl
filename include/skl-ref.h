// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// IWYU pragma: begin_exports

/*
 * GEMM Kernels
 */

#include "../ref/gemm/gemm_bf16rc_bf16rc_f32rc_scalar.h"
#include "../ref/gemm/gemm_f16rc_f16rc_f16rc_scalar.h"
#include "../ref/gemm/gemm_f16rc_f16rc_f32rc_scalar.h"
#include "../ref/gemm/gemm_f32rc_f32rc_f32rc_scalar.h"
#include "../ref/gemm/gemm_f64rc_f64rc_f64rc_scalar.h"
#include "../ref/gemm/gemm_f8e4m3rc_f8e4m3rc_f32rc_scalar.h"
#include "../ref/gemm/gemm_i8rc_i8rc_i32rc_scalar.h"

#include "../ref/gemm/gemm_bf16rcprc_bf16rcprc_f32rcprc_scalar.h"
#include "../ref/gemm/gemm_f16rcprc_f16rcprc_f32rcprc_scalar.h"
#include "../ref/gemm/gemm_f32rcprc_f32rcprc_f32rcprc_scalar.h"
#include "../ref/gemm/gemm_f64rcprc_f64rcprc_f64rcprc_scalar.h"
#include "../ref/gemm/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc_scalar.h"
#include "../ref/gemm/gemm_i8rcprc_i8rcprc_i32rcprc_scalar.h"

/*
 * Transpose Kernels
 */

#include "../ref/transpose/transpose_e16_scalar.h"
#include "../ref/transpose/transpose_e32_scalar.h"
#include "../ref/transpose/transpose_e8_scalar.h"

/*
 * Exponential and Logistic Function Kernels
 */

#include "../ref/logistic/logistic_f16_scalar.h"
#include "../ref/logistic/logistic_f32_scalar.h"

/*
 * Conversion Functions
 */
#include "../ref/cvt/cvt_f4e2m1_f8e4m3_scalar.h"
#include "../ref/cvt/cvt_ofp8_scalar.h"

/*
 * Depthwise Convolution Kernels
 */

#include "../ref/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_scalar.h"

#include "../ref/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_scalar.h"
#include "../ref/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_scalar.h"

/*
 * Softmax Kernels
 */

#include "../ref/softmax/softmax_bf16_scalar.h"
#include "../ref/softmax/softmax_f16_scalar.h"
#include "../ref/softmax/softmax_f32_scalar.h"

// IWYU pragma: end_exports
