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
  void *func; // polymorphic pointer to function
  char *name; // name of the function under test
  size_t n;   // input size
  __bf16 beta;

  // Buffer for input data
  SKL_TEST_BUFFER(__bf16) a;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *S; // reference result
    __bf16 *s; // result
    float max_err;
  } ctx;
} softmax_bf16_t;

void softmax_bf16_init(skl_test_t *);
void softmax_bf16_execute(skl_test_t *);
void softmax_bf16_verify(skl_test_t *);
void softmax_bf16_report(skl_test_t *);
void softmax_bf16_cleanup(skl_test_t *);

#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = softmax_bf16_init,                                               \
      .report = softmax_bf16_report,                                           \
      .cleanup = softmax_bf16_cleanup,                                         \
  }

#define TEST                                                                   \
  .a.mode = SKL_TEST_RANDOM, BASIC_STEPS, .steps.verify = softmax_bf16_verify

#define BENCHMARK .a.mode = SKL_TEST_SEQ, BASIC_STEPS, .steps.verify = NULL

// clang-format off
#define BASE_PARAMS(VAR)                                                       \
  .func = (void *)skl_softmax_bf16_##VAR,                                      \
  .name = "skl_softmax_bf16_" #VAR,                                            \
  .steps.execute = softmax_bf16_execute

#define BETA_BENCHMARKS(...)                                                   \
  {__VA_ARGS__, .beta = 1.0f},                                                 \
  {__VA_ARGS__, .beta = 1.1f}

#define VARIANT_BENCHMARKS(VAR)                                                \
  BETA_BENCHMARKS(BENCHMARK, BASE_PARAMS(VAR), .n = 2048)


#define BETA_TESTS(...)                                                        \
  {__VA_ARGS__, .beta = 0x1.62p-1},                                            \
  {__VA_ARGS__, .beta = 1.0f},                                                 \
  {__VA_ARGS__, .beta = 0x1.0ap+0}

#define DOMAIN_TESTS(...)                                                      \
  BETA_TESTS(__VA_ARGS__, .a.min =  -1, .a.max =   +1),                        \
  BETA_TESTS(__VA_ARGS__, .a.min = +40, .a.max = +100)

#define SIZE_TESTS(...)                                                        \
  DOMAIN_TESTS(__VA_ARGS__, .n = 1),                                           \
  DOMAIN_TESTS(__VA_ARGS__, .n = 27),                                          \
  DOMAIN_TESTS(__VA_ARGS__, .n = 131),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 380),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 732),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 1312)                                         \

#define VARIANT_TESTS(VAR)                                                     \
  SIZE_TESTS(TEST, BASE_PARAMS(VAR))

// clang-format on
