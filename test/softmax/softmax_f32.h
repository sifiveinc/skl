// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>

typedef struct {
  // Test function pointers for various steps
  skl_test_steps_t steps;

  // Configurable parameters (Softmax function and arguments)
  void *func;  // polymorphic pointer to function
  char *name;  // name of the function under test
  size_t m, n; // row and column count
  float beta;
  size_t rsa, rss; // row strides of input and output

  // Buffer for input data
  SKL_TEST_BUFFER(float) a;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *S; // reference result
    float *s;  // result
    float max_err;
  } ctx;
} softmax_f32_t;

void softmax_f32_init(skl_test_t *);
void softmax_f32_execute(skl_test_t *);
void softmax_2d_f32_execute(skl_test_t *);
void softmax_f32_verify(skl_test_t *);
void softmax_f32_report(skl_test_t *);
void softmax_f32_cleanup(skl_test_t *);

#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = softmax_f32_init,                                                \
      .report = softmax_f32_report,                                            \
      .cleanup = softmax_f32_cleanup,                                          \
  }

#define TEST                                                                   \
  .a.mode = SKL_TEST_RANDOM, BASIC_STEPS, .steps.verify = softmax_f32_verify

#define BENCHMARK .a.mode = SKL_TEST_SEQ, BASIC_STEPS, .steps.verify = NULL

// clang-format off
#define BASE_PARAMS(VAR)                                                       \
  .func = (void *)skl_softmax_f32_##VAR,                                       \
  .name = "skl_softmax_f32_" #VAR,                                             \
  .steps.execute = softmax_f32_execute

#define BASE_2D_PARAMS(VAR)                                                    \
  .func = (void *)skl_softmax_2d_f32_##VAR,                                    \
  .name = "skl_softmax_2d_f32_" #VAR,                                          \
  .steps.execute = softmax_2d_f32_execute


#define BETA_BENCHMARKS(...)                                                   \
  {__VA_ARGS__, .beta = 1.0f},                                                 \
  {__VA_ARGS__, .beta = 1.1f}

#define VARIANT_BENCHMARKS(VAR)                                                \
  BETA_BENCHMARKS(BENCHMARK, BASE_PARAMS(VAR), .m = 1, .n = 1024)

#define VARIANT_2D_BENCHMARKS(VAR)                                             \
  BETA_BENCHMARKS(BENCHMARK, BASE_2D_PARAMS(VAR), .m = 128, .n = 128)


#define BETA_TESTS(...)                                                        \
  {__VA_ARGS__, .beta = 0x1.62e43p-1f},                                        \
  {__VA_ARGS__, .beta = 1.0f},                                                 \
  {__VA_ARGS__, .beta = 0x1.0a2b24p0f}

#define DOMAIN_TESTS(...)                                                      \
  BETA_TESTS(__VA_ARGS__, .a.min =  -1, .a.max =   +1),                        \
  BETA_TESTS(__VA_ARGS__, .a.min = +40, .a.max = +100)

#define ALIGN(N) (((N) / 128) * 128 + 128)

#define NAT_TESTS(N,...)                                                       \
  DOMAIN_TESTS(__VA_ARGS__, .n = N)

#define ALIGN_TESTS(N,...)                                                     \
  NAT_TESTS(N,__VA_ARGS__),                                                    \
  DOMAIN_TESTS(__VA_ARGS__, .n = N, .rss = ALIGN(N), .rsa = ALIGN(N))

#define COLS_TESTS(CONTINUE,...)                                               \
  CONTINUE(1, __VA_ARGS__),                                                    \
  CONTINUE(27, __VA_ARGS__),                                                   \
  CONTINUE(131, __VA_ARGS__),                                                  \
  CONTINUE(380, __VA_ARGS__),                                                  \
  CONTINUE(732, __VA_ARGS__),                                                  \
  CONTINUE(1312, __VA_ARGS__)

#define SIZE_TESTS(...)                                                        \
  COLS_TESTS(NAT_TESTS,__VA_ARGS__, .m = 1)

#define ROWS_TESTS(...)                                                        \
  COLS_TESTS(ALIGN_TESTS,__VA_ARGS__, .m = 1),                                 \
  COLS_TESTS(ALIGN_TESTS,__VA_ARGS__, .m = 12),                                \
  COLS_TESTS(ALIGN_TESTS,__VA_ARGS__, .m = 37),                                \
  COLS_TESTS(ALIGN_TESTS,__VA_ARGS__, .m = 135)

#define ZERO_TESTS(...)                                                        \
  {__VA_ARGS__, .m = 0, .n = 8, .beta = 1.0f},                                 \
  {__VA_ARGS__, .m = 8, .n = 0, .beta = 1.0f},                                 \
  ROWS_TESTS(__VA_ARGS__)

#define VARIANT_TESTS(VAR)                                                     \
  SIZE_TESTS(TEST, BASE_PARAMS(VAR))

#define VARIANT_2D_TESTS(VAR)                                                  \
  ZERO_TESTS(TEST, BASE_2D_PARAMS(VAR))

// clang-format on
