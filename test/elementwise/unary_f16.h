// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Test and benchmark for FP16 unary functions f(x) = y.
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
typedef void (*unary_func_f16_t)(_Float16 *, const _Float16 *, size_t);

// Type of SKL unary function harnesses
typedef struct {
  // Test function pointers for various steps
  skl_test_steps_t steps;

  // Configurable parameters
  const char *func_name;     // The name of the function to test
  unary_func_f16_t func;     // The function to test
  unary_func_f16_t ref_func; // The reference function

  // Buffer generation settings
  SKL_TEST_BUFFER(_Float16) in; // Input buffer

  // Derived parameters and buffers (private to the test harness)
  struct {
    _Float16 *out;     // Result buffer
    _Float16 *ref;     // Reference buffer
    float max_err;     // Ulp tolerance
    size_t max_errors; // Maximum number of errors to report
  } ctx;
} unary_f16_t;

void unary_f16_init(skl_test_t *t);
void unary_f16_execute(skl_test_t *t);
void unary_f16_verify(skl_test_t *t);
void unary_f16_report(skl_test_t *t);
void unary_f16_cleanup(skl_test_t *t);

#define UNARY_FP16_TEST_DEFAULTS .in = {.len = 16384, .mode = SKL_TEST_RANDOM}
#define UNARY_FP16_BENCH_DEFAULTS .in = {.len = 1024, .mode = SKL_TEST_SEQ}
#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = unary_f16_init,                                                  \
      .execute = unary_f16_execute,                                            \
      .cleanup = unary_f16_cleanup,                                            \
      .report = unary_f16_report,                                              \
  }

#define TEST                                                                   \
  UNARY_FP16_TEST_DEFAULTS, BASIC_STEPS, .steps.warmup = NULL,                 \
                                         .steps.verify = unary_f16_verify,     \
                                         .ctx.max_errors = 10

#define BENCH                                                                  \
  UNARY_FP16_BENCH_DEFAULTS, BASIC_STEPS, .steps.warmup = unary_f16_execute,   \
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
