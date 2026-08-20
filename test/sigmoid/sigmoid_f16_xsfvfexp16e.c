// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

/**
 * @brief Consolidated FP16 sigmoid-family test cases for xsfvfexp16e.
 *
 * Exercises the logistic, SiLU, Swish, GLU, and SwiGLU kernels through the
 * unary_f16 harness, one entry per API in a single suite.
 */

#include <stddef.h>

#include "elementwise/unary_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

// Per-API input ranges and tuning parameters.
#define LOGISTIC_MIN (-18.f)
#define LOGISTIC_MAX (+10.f)
#define SILU_MIN (-21.f)
#define SILU_MAX (+9.f)
#define SWISH_MIN (-21.f)
#define SWISH_MAX (+9.f)
#define SWISH_BETA (_Float16)2
#define GLU_MIN (-21.f)
#define GLU_MAX (+9.f)
#define SWIGLU_MIN (-11.f)
#define SWIGLU_MAX (+9.f)
#define SWIGLU_DELTA (_Float16)0.5

void test_swish_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n) {
  skl_swish_f16_xsfvfexp16e(out, SWISH_BETA, in, n);
}

void ref_swish_f16(_Float16 *out, const _Float16 *in, size_t n) {
  skl_swish_f16_ref(out, SWISH_BETA, in, n);
}

void test_glu_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n) {
  skl_glu_f16_xsfvfexp16e(out, in, in, n);
}

void ref_glu_f16(_Float16 *out, const _Float16 *in, size_t n) {
  skl_glu_f16_ref(out, in, in, n);
}

void test_swiglu_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n) {
  skl_swiglu_f16_xsfvfexp16e(out, in, in, SWIGLU_DELTA, n);
}

void ref_swiglu_f16(_Float16 *out, const _Float16 *in, size_t n) {
  skl_swiglu_f16_ref(out, in, in, SWIGLU_DELTA, n);
}

unary_f16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_logistic_f16_xsfvfexp16e, -1, 1),
    FUNCTION_BENCHMARKS(skl_silu_f16_xsfvfexp16e, -1, 1),
    FUNCTION_BENCHMARKS(test_swish_f16_xsfvfexp16e, -1, 1),
    FUNCTION_BENCHMARKS(test_glu_f16_xsfvfexp16e, -1, 1),
    FUNCTION_BENCHMARKS(test_swiglu_f16_xsfvfexp16e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_logistic_f16_xsfvfexp16e, skl_logistic_f16_ref,
                   LOGISTIC_MIN, LOGISTIC_MAX, 5.0f),
    FUNCTION_TESTS(skl_silu_f16_xsfvfexp16e, skl_silu_f16_ref, SILU_MIN,
                   SILU_MAX, 31.0f),
    FUNCTION_TESTS(test_swish_f16_xsfvfexp16e, ref_swish_f16, SWISH_MIN,
                   SWISH_MAX, 31.0f),
    FUNCTION_TESTS(test_glu_f16_xsfvfexp16e, ref_glu_f16, GLU_MIN, GLU_MAX,
                   31.0f),
    FUNCTION_TESTS(test_swiglu_f16_xsfvfexp16e, ref_swiglu_f16, SWIGLU_MIN,
                   SWIGLU_MAX, 31.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_sigmoid_f16_xsfvfexp16e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
