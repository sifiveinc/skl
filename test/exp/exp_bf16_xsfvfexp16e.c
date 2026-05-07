// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

/**
 * @brief Test cases for BF16 Exponential with Xsfvfexp16e extension.
 *
 * This test uses the unary_bf16 harness.
 */

#include "elementwise/unary_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-63.75f)
#define MAX (+63.75f)

unary_bf16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e, skl_exp_bf16_ref,
                   MIN, MAX, 1.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_bf16_xsfvfexp16e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_bf16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
