// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

/**
 * @brief Consolidated FP32 sigmoid-family test cases for xsfvfexp32e.
 *
 * Exercises the logistic, SiLU, Swish, GLU, and SwiGLU kernels through the
 * unary_f32 harness, one entry per API in a single suite.
 */

#include <stddef.h>

#include "elementwise/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

// Per-API input ranges and tuning parameters.
#define LOGISTIC_MIN (-104.f)
#define LOGISTIC_MAX (+18.f)
#define SILU_MIN (-110.f)
#define SILU_MAX (+18.f)
#define SWISH_MIN (-110.f)
#define SWISH_MAX (+18.f)
#define SWISH_BETA 2.f
#define GLU_MIN (-110.f)
#define GLU_MAX (+18.f)
#define SWIGLU_MIN (-55.f)
#define SWIGLU_MAX (+18.f)
#define SWIGLU_DELTA 0.5f

void test_swish_f32_xsfvfexp32e(float *out, const float *in, size_t n) {
  skl_swish_f32_xsfvfexp32e(out, SWISH_BETA, in, n);
}

void ref_swish_f32(float *out, const float *in, size_t n) {
  skl_swish_f32_ref(out, SWISH_BETA, in, n);
}

void test_glu_f32_xsfvfexp32e(float *out, const float *in, size_t n) {
  skl_glu_f32_xsfvfexp32e(out, in, in, n);
}

void ref_glu_f32(float *out, const float *in, size_t n) {
  skl_glu_f32_ref(out, in, in, n);
}

void test_swiglu_f32_xsfvfexp32e(float *out, const float *in, size_t n) {
  skl_swiglu_f32_xsfvfexp32e(out, in, in, SWIGLU_DELTA, n);
}

void ref_swiglu_f32(float *out, const float *in, size_t n) {
  skl_swiglu_f32_ref(out, in, in, SWIGLU_DELTA, n);
}

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_logistic_f32_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(skl_silu_f32_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_swish_f32_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_glu_f32_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_swiglu_f32_xsfvfexp32e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_logistic_f32_xsfvfexp32e, skl_logistic_f32_ref,
                   LOGISTIC_MIN, LOGISTIC_MAX, 8.0f),
    FUNCTION_TESTS(skl_silu_f32_xsfvfexp32e, skl_silu_f32_ref, SILU_MIN,
                   SILU_MAX, 52.0f),
    FUNCTION_TESTS(test_swish_f32_xsfvfexp32e, ref_swish_f32, SWISH_MIN,
                   SWISH_MAX, 52.0f),
    FUNCTION_TESTS(test_glu_f32_xsfvfexp32e, ref_glu_f32, GLU_MIN, GLU_MAX,
                   52.0f),
    FUNCTION_TESTS(test_swiglu_f32_xsfvfexp32e, ref_swiglu_f32, SWIGLU_MIN,
                   SWIGLU_MAX, 52.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_sigmoid_f32_xsfvfexp32e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
