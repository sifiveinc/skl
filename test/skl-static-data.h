// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <stddef.h>

#define SKL_ARRAY_LEN(ARR) sizeof((ARR)) / sizeof((ARR)[0])

static const __bf16 skl_static_nan_bf16[] = {
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
static const size_t skl_static_nan_bf16_len =
    SKL_ARRAY_LEN(skl_static_nan_bf16);

static const _Float16 skl_static_nan_f16[] = {
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
static const size_t skl_static_nan_f16_len = SKL_ARRAY_LEN(skl_static_nan_f16);

static const float skl_static_nan_f32[] = {
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
static const size_t skl_static_nan_f32_len = SKL_ARRAY_LEN(skl_static_nan_f32);

static const double skl_static_nan_f64[] = {
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
static const size_t skl_static_nan_f64_len = SKL_ARRAY_LEN(skl_static_nan_f64);

#undef SKL_ARRAY_LEN
