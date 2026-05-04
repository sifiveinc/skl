// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

/**
 * @brief Test cases for Exponential with Zve32f extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "elementwise/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-104.f)
#define MAX (+89.f)

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1u_f32_zve32f, MIN, MAX),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u_f32_zve32f, skl_exp_f32_ref, MIN, MAX, 1.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f32_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
