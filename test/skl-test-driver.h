// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
/**
 * @file skl-test.c
 * @brief Test framework for SKL functions.
 *
 * Example implementation:
 * @code
 *
 * typedef void (*gemm_test_func_t)(size_t, size_t, size_t, float, const float *,
 * size_t, const float *, size_t, size_t, float, float *, size_t, size_t);
 *
 * struct {
 *     size_t M;
 *     size_t N;
 *     size_t K;
 *     size_t RSA;
 *     size_t CSA;
 *     size_t RSB;
 *     size_t CSB;
 *     size_t RSC;
 *     size_t CSC;
 *     float ALPHA;
 *     float BETA;
 *     gemm_test_func_t FUNC;
 * } gemm_params;
 *
 * const struct { skl_test_param_t *params; size_t num_params; } default_tests = {
 *     .params = (skl_test_param_t[]) {
 *         { "M", "1024" },
 *         { "N", "1024" },
 *         { "K", "1024" },
 *         { "ALPHA", "1.0" },
 *         { "BETA", "0.0" },
 *         { "TEST_FUNC", "skl_gemm_f32_f32_f32_zve32f_x390_wrapper" },
 *         { "REF_FUNC", "skl_gemm_f32_f32_f32_ref" },
 *     },
 *     .num_params = 7,
 * };
 *
 * // The buffers to use for the test.
 * static float *A, *B, *C;
 *
 * int SKL_TEST_CONFIG(skl_test_param_t *params, size_t num_params) {
 *     // Parse the configurable parameters.
 *     SKL_TEST_PARAMS(params, num_params,
 *         SKL_TEST_PARAM_SZ ("M",        gemm_params.M),
 *         SKL_TEST_PARAM_SZ ("N",        gemm_params.N),
 *         SKL_TEST_PARAM_SZ ("K",        gemm_params.K),
 *         SKL_TEST_PARAM_F32("ALPHA",    gemm_params.ALPHA),
 *         SKL_TEST_PARAM_F32("BETA",     gemm_params.BETA)
 *     );
 *
 *     // Set the derived parameters.
 *     gemm_params.RSA = gemm_params.M;
 *     gemm_params.CSA = 1;
 *     gemm_params.RSB = gemm_params.N;
 *     gemm_params.CSB = 1;
 *     gemm_params.RSC = gemm_params.N;
 *     gemm_params.CSC = 1;
 *
 *     // Allocate and initialize the buffers.
 *     SKL_TEST_BUFFER(&A, float, gemm_params.M * gemm_params.K);
 *     SKL_TEST_BUFFER(&B, float, gemm_params.K * gemm_params.N);
 *     SKL_TEST_BUFFER(&C, float, gemm_params.M * gemm_params.N);
 *
 *     return 0;
 * }
 *
 * int SKL_TEST_EXECUTE(void) {
 *     // Execute the test.
 *     return gemm_params.FUNC(gemm_params.M, gemm_params.N, gemm_params.K,
 *                      gemm_params.ALPHA, A, gemm_params.RSA, gemm_params.CSA, B,
 *                      gemm_params.RSB, gemm_params.CSB,
 *                      gemm_params.BETA, C, gemm_params.RSC, gemm_params.CSC);
 * }
 * 
 * int SKL_TEST_REPORT(uint64_t cycles, uint64_t insts) {
 *     size_t maccs = gemm_params.M * gemm_params.N * gemm_params.K;
 *     float mpc = (float)maccs / (float)cycles;
 *     SKL_TEST_RESULT("MACCS", "%lu", maccs);
 *     SKL_TEST_RESULT("MACCS/CYCLE", "%f", mpc);
 *     return 0;
 * }
 *
 * int SKL_TEST_VERIFY(void) {
 *     // Verify the results unless this is a performance test.
 *     return 0;
 * }
 *
 * int SKL_TEST_FINISH(void) {
 *     // Free the buffers.
 *     SKL_TEST_FREE(A);
 *     SKL_TEST_FREE(B);
 *     SKL_TEST_FREE(C);
 *     return 0;
 * }
 *
 * @endcode
 */
// clang-format on

#include <stdio.h>

#define SKL_TEST_RESULT(NAME, FMT, ...)                                        \
  printf("%s: " FMT "\n", NAME, __VA_ARGS__)

#define SKL_TEST_LOG(...)                                                      \
  if (SKL_TEST_LOG_LEVEL >= 1) {                                               \
    printf("SKL TEST: ");                                                      \
    printf(__VA_ARGS__);                                                       \
  }

typedef enum {
  SKL_TEST_RANDOM,
  SKL_TEST_SEQ,
  SKL_TEST_STATIC,
} skl_test_data_t;

typedef struct {
  const char *name;
  (void *)(*memalign)(size_t, size_t);
  void (*free)(void *);
  size_t alignment;
  bool warm_cache;
  bool verify;
  skl_test_data_t data;
  float float_min;
  float float_max;
  int32_t int32_min;
  int32_t int32_max;
  int8_t int8_min;
  int8_t int8_max;
} skl_test_config_t;

/**
 * @brief The test configuration to be set by each test.
 *
 * During the configuration phase, the SKL_TEST_CONFIG function should parse
 * the parameters and set the values of this structure accordingly. The test
 * driver initializes it with default values, which can be overridden by the
 * test.
 */
extern skl_test_config_t skl_test_config;

#define SKL_TEST_RESERVED_PARAMS                                               \
  SKL_TEST_PARAM_STR("NAME", skl_test_config.name)                             \
  SKL_TEST_PARAM_SZ("ALIGNMENT", skl_test_config.alignment)                    \
  SKL_TEST_PARAM_BOOL("WARM_CACHE", skl_test_config.warm_cache)                \
  SKL_TEST_PARAM_BOOL("VERIFY", skl_test_config.verify)                        \
  SKL_TEST_PARAM_ENUM("DATA", skl_test_config.data, "RANDOM", "SEQ", "STATIC") \
  SKL_TEST_PARAM_F32("FLOAT_MIN", skl_test_config.float_min)                   \
  SKL_TEST_PARAM_F32("FLOAT_MAX", skl_test_config.float_max)                   \
  SKL_TEST_PARAM_I32("INT32_MIN", skl_test_config.int32_min)                   \
  SKL_TEST_PARAM_I32("INT32_MAX", skl_test_config.int32_max)                   \
  SKL_TEST_PARAM_I8("INT8_MIN", skl_test_config.int8_min)                      \
  SKL_TEST_PARAM_I8("INT8_MAX", skl_test_config.int8_max)

typedef struct {
  const char *param;
  const char *value;
} skl_test_param_t;

#define SKL_TEST_PARAM_(NAME, VAR, TYPE, PARSE, FMT)
else if (strcmp(params[i].param, NAME) == 0) {
  VAR = (TYPE)PARSE(params[i].value);
  SKL_TEST_LOG("Set %s to %" FMT "\n", NAME, VAR);
}

#define SKL_TEST_PARAM_STR(NAME, VAR)                                          \
  SKL_TEST_PARAM_(NAME, VAR, const char *, strdup, "%s")

#define SKL_TEST_PARAM_SZ(NAME, VAR)                                           \
  SKL_TEST_PARAM_(NAME, VAR, size_t, atoll, "lu")

#define SKL_TEST_PARAM_I8(NAME, VAR)                                           \
  SKL_TEST_PARAM_(NAME, VAR, int8_t, atoi, "d")

#define SKL_TEST_PARAM_I32(NAME, VAR)                                          \
  SKL_TEST_PARAM_(NAME, VAR, int32_t, atoi, "d")

#define SKL_TEST_PARAM_F32(NAME, VAR)                                          \
  SKL_TEST_PARAM_(NAME, VAR, float, atof, "f")

#define SKL_TEST_PARAMS(PARAMS, NUM_PARAMS, ...)                               \
  do {                                                                         \
    skl_test_param_t *params = (PARAMS);                                       \
    for (size_t i = 0; i < (NUM_PARAMS); ++i) {                                \
      if (0) {                                                                 \
      }                                                                        \
      SKL_TEST_RESERVED_PARAMS                                                 \
      __VA_ARGS__                                                              \
      else {                                                                   \
        fprintf(stderr, "Unknown parameter: %s\n", params[i].param);           \
        return 1;                                                              \
      }                                                                        \
    }                                                                          \
  } while (0)

#define SKL_TEST_INIT_(CFG, TYPE, BUF, NUM, RAND)                              \
  do {                                                                         \
    TYPE *buf = (BUF);                                                         \
    const size_t num = (NUM);                                                  \
    switch (CFG.data) {                                                        \
    case SKL_TEST_RANDOM:                                                      \
      for (size_t i = 0; i < num; ++i) {                                       \
        buf[i] = RAND;                                                         \
      }                                                                        \
      break;                                                                   \
    case SKL_TEST_SEQ:                                                         \
      for (size_t i = 0; i < num; ++i) {                                       \
        buf[i] = (TYPE)(min + step * i);                                       \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
  case SKL_TEST_STATIC:                                                        \
    for (size_t i = 0; i < num; ++i) {                                         \
      buf[i] =                                                                 \
          SKL_TEST_STATIC_DATA_NAME(BUF)[i % SKL_TEST_STATIC_DATA_LEN(BUF)];   \
      \                                                                        \
    }                                                                          \
    break;                                                                     \
  }
}
while (0)

#define SKL_TEST_INIT__Float16(CFG, BUF, NUM)                                  \
  SKL_TEST_INIT_(CFG, _Float16, BUF, NUM,                                      \
                 ((float)rand() / (float)RAND_MAX) *                           \
                         (CFG.float_max - CFG.float_min) +                     \
                     CFG.float_min)

#define SKL_TEST_INIT_float(CFG, BUF, NUM)                                     \
  SKL_TEST_INIT_(CFG, float, BUF, NUM,                                         \
                 ((float)rand() / (float)RAND_MAX) *                           \
                         (CFG.float_max - CFG.float_min) +                     \
                     CFG.float_min)

#define SKL_TEST_INIT_double(CFG, BUF, NUM)                                    \
  SKL_TEST_INIT_(CFG, double, BUF, NUM,                                        \
                 ((double)rand() / (double)RAND_MAX) *                         \
                         (CFG.float_max - CFG.float_min) +                     \
                     CFG.float_min)

#define SKL_TEST_INIT_int8_t(CFG, BUF, NUM)                                    \
  SKL_TEST_INIT_(CFG, int8_t, BUF, NUM,                                        \
                 (int8_t)rand() % (CFG.int8_max - CFG.int8_min + 1) +          \
                     CFG.int8_min)

#define SKL_TEST_INIT_int32_t(CFG, BUF, NUM)                                   \
  SKL_TEST_INIT_(CFG, int32_t, BUF, NUM,                                       \
                 (int32_t)rand() % (CFG.int32_max - CFG.int32_min + 1) +       \
                     CFG.int32_min)

#define SKL_TEST_BUFFER(NAME, TYPE, NUM)                                       \
  *NAME = (TYPE *)SKL_TEST_MEMALIGN(SKL_TEST_ALIGNMENT, (NUM) * sizeof(TYPE)); \
  SKL_TEST_BUFFER_INIT_##TYPE(NAME, NUM)