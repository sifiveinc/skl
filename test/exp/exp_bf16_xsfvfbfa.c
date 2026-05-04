// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

/**
 * @brief Test cases for BF16 Exponential with Xsfvfbda extension.
 *
 * This test uses the unary_bf16 harness.
 */

#include "elementwise/unary_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-104.f)
#define MAX (+89.f)

unary_bf16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1u_bf16_xsfvfbfa, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u_bf16_xsfvfbfa, skl_exp_bf16_ref, MIN, MAX, 1.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_bf16_xsfvfbfa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_bf16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
