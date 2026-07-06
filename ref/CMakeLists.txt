# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

# Build the reference library

add_library(skl-ref
  STATIC
  EXCLUDE_FROM_ALL
    cvt/cvt_ofp4.c
    cvt/cvt_ofp8.c
    depthwise_conv2d/depthwise_conv2d_f16_f16_f16.c
    depthwise_conv2d/depthwise_conv2d_f32_f32_f32.c
    depthwise_conv2d/depthwise_conv2d_f64_f64_f64.c
    depthwise_conv2d/depthwise_conv2d_i8_i8_i32.c
    exp/exp_bf16.c
    exp/exp_f16.c
    exp/exp_f32.c
    gelu/gelu_f32.c
    gemm/gemm_bf16rc_bf16rc_f32rc.c
    gemm/gemm_bf16rcprc_bf16rcprc_f32rcprc.c
    gemm/gemm_f8e4m3rc_f8e4m3rc_f32rc.c
    gemm/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc.c
    gemm/gemm_f8e5m2rc_f8e5m2rc_f32rc.c
    gemm/gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc.c
    gemm/gemm_f16rc_f16rc_f16rc.c
    gemm/gemm_f16rcprc_f16rcprc_f16rcprc.c
    gemm/gemm_f16rc_f16rc_f32rc.c
    gemm/gemm_f16rcprc_f16rcprc_f32rcprc.c
    gemm/gemm_f32rc_f32rc_f32rc.c
    gemm/gemm_f32rcprc_f32rcprc_f32rcprc.c
    gemm/gemm_f64rc_f64rc_f64rc.c
    gemm/gemm_f64rcprc_f64rcprc_f64rcprc.c
    gemm/gemm_i8rc_i8rc_i32rc.c
    gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
    logistic/logistic_bf16.c
    logistic/logistic_f16.c
    logistic/logistic_f32.c
    pack/pack_e16rc_e16rcprc.c
    pack/pack_e32rc_e32rcprc.c
    pack/pack_e8rc_e8rcprc.c
    rmsnorm/rmsnorm_f32.c
    silu/silu_f16.c
    silu/silu_f32.c
    softmax/softmax_bf16.c
    softmax/softmax_f16.c
    softmax/softmax_f32.c
    transpose/transpose_e16.c
    transpose/transpose_e32.c
    transpose/transpose_e8.c
)

target_include_directories(skl-ref PUBLIC ../include ..)
target_compile_options(skl-ref PRIVATE ${SKL_COMPILE_OPTIONS})
target_link_libraries(skl-ref PUBLIC m)
