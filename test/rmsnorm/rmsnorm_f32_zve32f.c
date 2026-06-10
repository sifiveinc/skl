// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

/**
 * @brief Test cases for RMSNorm with Zve32f extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "rmsnorm/rmsnorm_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

rmsnorm_f32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    VARIANT_BENCHMARKS(zve32f),
#endif
// #ifdef SKL_ENABLE_TESTS
//     VARIANT_TESTS(zve32f),
// #endif
};

static skl_test_suite_t suite = {.name = "skl_rmsnorm_f32_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(rmsnorm_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
