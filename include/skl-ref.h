// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

// IWYU pragma: begin_exports

/*
 * GEMM Kernels
 */

#include "../ref/gemm/gemm_bf16rc_bf16rc_f32rc.h"
#include "../ref/gemm/gemm_f16rc_f16rc_f16rc.h"
#include "../ref/gemm/gemm_f16rc_f16rc_f32rc.h"
#include "../ref/gemm/gemm_f32rc_f32rc_f32rc.h"
#include "../ref/gemm/gemm_f64rc_f64rc_f64rc.h"
#include "../ref/gemm/gemm_f8e4m3rc_f8e4m3rc_f32rc.h"
#include "../ref/gemm/gemm_f8e5m2rc_f8e5m2rc_f32rc.h"
#include "../ref/gemm/gemm_i8rc_i8rc_i32rc.h"

#include "../ref/gemm/gemm_bf16rcprc_bf16rcprc_f32rcprc.h"
#include "../ref/gemm/gemm_f16rcprc_f16rcprc_f16rcprc.h"
#include "../ref/gemm/gemm_f16rcprc_f16rcprc_f32rcprc.h"
#include "../ref/gemm/gemm_f32rcprc_f32rcprc_f32rcprc.h"
#include "../ref/gemm/gemm_f64rcprc_f64rcprc_f64rcprc.h"
#include "../ref/gemm/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc.h"
#include "../ref/gemm/gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc.h"
#include "../ref/gemm/gemm_i8rcprc_i8rcprc_i32rcprc.h"

/*
 * Transpose Kernels
 */

#include "../ref/transpose/transpose_e16.h"
#include "../ref/transpose/transpose_e32.h"
#include "../ref/transpose/transpose_e8.h"

/*
 * Exponential Function Kernels
 */

#include "../ref/exp/exp_bf16.h"
#include "../ref/exp/exp_f16.h"
#include "../ref/exp/exp_f32.h"

/*
 * GELU Function Kernels
 */

#include "../ref/gelu/gelu_f32.h"

/*
 * Logistic Function Kernels
 */

#include "../ref/logistic/logistic_bf16.h"
#include "../ref/logistic/logistic_f16.h"
#include "../ref/logistic/logistic_f32.h"

/*
 * SiLU Function Kernels
 */

#include "../ref/silu/silu_f16.h"
#include "../ref/silu/silu_f32.h"

/*
 * Conversion Functions
 */
#include "../ref/cvt/cvt_ofp4.h"
#include "../ref/cvt/cvt_ofp8.h"

/*
 * Depthwise Convolution Kernels
 */

#include "../ref/depthwise_conv2d/depthwise_conv2d_f16_f16_f16.h"

#include "../ref/depthwise_conv2d/depthwise_conv2d_f32_f32_f32.h"
#include "../ref/depthwise_conv2d/depthwise_conv2d_f64_f64_f64.h"
#include "../ref/depthwise_conv2d/depthwise_conv2d_i8_i8_i32.h"

/*
 * Softmax Kernels
 */

#include "../ref/softmax/softmax_bf16.h"
#include "../ref/softmax/softmax_f16.h"
#include "../ref/softmax/softmax_f32.h"

/*
 * Pack Kernels
 */

#include "../ref/pack/pack_e16rc_e16rcprc.h"
#include "../ref/pack/pack_e32rc_e32rcprc.h"
#include "../ref/pack/pack_e8rc_e8rcprc.h"

// IWYU pragma: end_exports
