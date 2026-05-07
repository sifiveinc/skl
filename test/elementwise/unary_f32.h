// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for FP32 unary functions f(x) = y.
 *
 * This test uses a table-driven approach where test configurations
 * are defined in the `tests` array.  Each test specifies:
 *  - Array dimension
 *  - Expected maximum numerical error of the result
 *  - The kernel function to test
 */

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

// Type of SKL unary functions
typedef void (*unary_func_f32_t)(float *, const float *, size_t);

// Type of SKL unary function harnesses
typedef struct {
  // Test function pointers for various steps
  skl_test_steps_t steps;

  // Configurable parameters
  const char *func_name;     // The name of the function to test
  unary_func_f32_t func;     // The function to test
  unary_func_f32_t ref_func; // The reference function

  // Buffer generation settings
  SKL_TEST_BUFFER(float) in; // Input buffer

  // Derived parameters and buffers (private to the test harness)
  struct {
    float *out;        // Result buffer
    float *ref;        // Reference buffer
    float max_err;     // Ulp tolerance
    size_t max_errors; // Maximum number of errors to report
  } ctx;
} unary_f32_t;

void unary_f32_init(skl_test_t *t);
void unary_f32_execute(skl_test_t *t);
void unary_f32_verify(skl_test_t *t);
void unary_f32_report(skl_test_t *t);
void unary_f32_cleanup(skl_test_t *t);

#define UNARY_F32_TEST_DEFAULTS .in = {.len = 65536, .mode = SKL_TEST_RANDOM}
#define UNARY_F32_BENCH_DEFAULTS .in = {.len = 1024, .mode = SKL_TEST_SEQ}
#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = unary_f32_init,                                                  \
      .execute = unary_f32_execute,                                            \
      .cleanup = unary_f32_cleanup,                                            \
      .report = unary_f32_report,                                              \
  }

#define TEST                                                                   \
  UNARY_F32_TEST_DEFAULTS, BASIC_STEPS, .steps.warmup = NULL,                  \
                                        .steps.verify = unary_f32_verify,      \
                                        .ctx.max_errors = 10

#define BENCH                                                                  \
  UNARY_F32_BENCH_DEFAULTS, BASIC_STEPS, .steps.warmup = unary_f32_execute,    \
                                         .steps.verify = NULL

#define FUNCTION_TESTS(FUN, REF_FUN, MIN, MAX, ULP)                            \
  {TEST,                                                                       \
   .func_name = #FUN,                                                          \
   .func = FUN,                                                                \
   .ref_func = REF_FUN,                                                        \
   .in.min = (MIN),                                                            \
   .in.max = (MAX),                                                            \
   .ctx.max_err = (ULP)}

#define FUNCTION_BENCHMARKS(FUN, MIN, MAX)                                     \
  {BENCH, .func_name = #FUN, .func = FUN, .in.min = (MIN), .in.max = (MAX)}
