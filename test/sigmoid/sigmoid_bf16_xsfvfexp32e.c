// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

/**
 * @brief Consolidated bf16 sigmoid-family test cases for xsfvfexp32e.
 *
 * Exercises the logistic, SiLU, Swish, GLU, and SwiGLU kernels through the
 * unary_bf16 harness, one entry per API in a single suite.
 */

#include <stddef.h>

#include "elementwise/unary_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

// Per-API input ranges and tuning parameters.
#define LOGISTIC_MIN (-0x1.74p6f)
#define LOGISTIC_MAX (+0x1.90p2f)
#define SILU_MIN (-40.f)
#define SILU_MAX (+6.f)
#define SWISH_MIN (-40.f)
#define SWISH_MAX (+6.f)
#define SWISH_BETA_POS (__bf16)2
#define SWISH_BETA_ONE (__bf16)1
#define SWISH_BETA_NEG (__bf16)-2
#define GLU_MIN (-40.f)
#define GLU_MAX (+6.f)
#define SWIGLU_MIN (-40.f)
#define SWIGLU_MAX (+6.f)
#define SWIGLU_DELTA (__bf16)0.5

static void test_swish_b1_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in,
                                           size_t n) {
  skl_swish_bf16_xsfvfexp32e(out, SWISH_BETA_ONE, in, n);
}

static void ref_swish_b1_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  skl_swish_bf16_ref(out, SWISH_BETA_ONE, in, n);
}

static void test_swish_bp_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in,
                                           size_t n) {
  skl_swish_bf16_xsfvfexp32e(out, SWISH_BETA_POS, in, n);
}

static void ref_swish_bp_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  skl_swish_bf16_ref(out, SWISH_BETA_POS, in, n);
}

static void test_swish_bn_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in,
                                           size_t n) {
  skl_swish_bf16_xsfvfexp32e(out, SWISH_BETA_NEG, in, n);
}

static void ref_swish_bn_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  skl_swish_bf16_ref(out, SWISH_BETA_NEG, in, n);
}

static void test_glu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in, size_t n) {
  skl_glu_bf16_xsfvfexp32e(out, in, in, n);
}

static void ref_glu_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  skl_glu_bf16_ref(out, in, in, n);
}

static void test_swiglu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in,
                                         size_t n) {
  skl_swiglu_bf16_xsfvfexp32e(out, in, in, SWIGLU_DELTA, n);
}

static void ref_swiglu_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  skl_swiglu_bf16_ref(out, in, in, SWIGLU_DELTA, n);
}

unary_bf16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_logistic_bf16_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(skl_silu_bf16_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_swish_bp_bf16_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_glu_bf16_xsfvfexp32e, -1, 1),
    FUNCTION_BENCHMARKS(test_swiglu_bf16_xsfvfexp32e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_logistic_bf16_xsfvfexp32e, skl_logistic_bf16_ref,
                   LOGISTIC_MIN, LOGISTIC_MAX, 2.0f),
    FUNCTION_TESTS(skl_silu_bf16_xsfvfexp32e, skl_silu_bf16_ref, SILU_MIN,
                   SILU_MAX, 2.0f),
    FUNCTION_TESTS(test_swish_b1_bf16_xsfvfexp32e, ref_swish_b1_bf16, SWISH_MIN,
                   SWISH_MAX, 2.0f),
    FUNCTION_TESTS(test_swish_bp_bf16_xsfvfexp32e, ref_swish_bp_bf16, SWISH_MIN,
                   SWISH_MAX, 2.0f),
    FUNCTION_TESTS(test_swish_bn_bf16_xsfvfexp32e, ref_swish_bn_bf16, SWISH_MIN,
                   SWISH_MAX, 2.0f),
    FUNCTION_TESTS(test_glu_bf16_xsfvfexp32e, ref_glu_bf16, GLU_MIN, GLU_MAX,
                   2.0f),
    FUNCTION_TESTS(test_swiglu_bf16_xsfvfexp32e, ref_swiglu_bf16, SWIGLU_MIN,
                   SWIGLU_MAX, 2.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_sigmoid_bf16_xsfvfexp32e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_bf16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
