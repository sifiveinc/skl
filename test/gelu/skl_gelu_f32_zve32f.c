// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

/**
 * @brief Test cases for GELU with Zve32f extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "elementwise/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-0x1.cb64e8p3f) // ~14.356
#define MAX (+0x1.563db2p2f) // ~5.348

// clang-format off
#define VARIANT_TESTS(VARIANT, MIN, MAX, ERR)                                  \
  FUNCTION_TESTS(skl_gelu_##VARIANT##_f32_zve32f, skl_gelu_f32_ref, MIN, MAX, ERR)

#define GELU_TESTS(VARIANT, LTN2, GEN2, GEN1, GE0)                             \
  VARIANT_TESTS(VARIANT, MIN,  -2, LTN2),                                      \
  VARIANT_TESTS(VARIANT,  -2,  -1, GEN2),                                      \
  VARIANT_TESTS(VARIANT,  -1,  -0, GEN1),                                      \
  VARIANT_TESTS(VARIANT,  +0, MAX, GE0)

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_gelu_p9_f32_zve32f, MIN, MAX),
    FUNCTION_BENCHMARKS(skl_gelu_p13_f32_zve32f, MIN, MAX),
    FUNCTION_BENCHMARKS(skl_gelu_p17_f32_zve32f, MIN, MAX),
    FUNCTION_BENCHMARKS(skl_gelu_rat_f32_zve32f, MIN, MAX),
#endif
#if defined(SKL_ENABLE_TESTS)
    GELU_TESTS(p9,  1.7e7f, 540097.f, 93837.f, 37021.f),
    GELU_TESTS(p13, 1.7e7f,  69077.f,  6810.f,  2299.f),
    GELU_TESTS(p17, 1.7e7f,   2732.f,   239.f,   239.f),
    GELU_TESTS(rat, 1.7e7f,     62.f,     5.f,     4.f)
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gelu_f32_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
