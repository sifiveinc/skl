// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/* Test and benchmark for SKL kernels.
 */

#include "skl-test.h"

typedef enum {
    SKL_TEST_RANDOM,
    SKL_TEST_SEQ,
    SKL_TEST_STATIC,
} skl_test_data_t;

typedef struct {
    (void *)(*memalign)(size_t, size_t);
    void (*free)(void *);
    size_t alignment;
    skl_test_data_t data;
    float float_min;
    float float_max;
    int32_t int32_min;
    int32_t int32_max;
    int8_t int8_min;
    int8_t int8_max;
} skl_test_config_t;

static skl_test_config_t skl_test_config_default = {
    .memalign = aligned_alloc,
    .free = free,
    .alignment = 64,
    .data = SKL_TEST_RANDOM,
    .float_min = -10.0f,
    .float_max = 10.0f,
    .int32_min = -100,
    .int32_max = 100,
    .int8_min = -128,
    .int8_max = 127,
};

typedef struct {
    const char *param;
    const char *value;
} skl_test_param_t;

#define SKL_TEST_PARAM_SZ(NAME, VAR)                                          \
    else if (strcmp(params[i].param, #NAME) == 0) {                            \
        VAR = (size_t)atoll(params[i].value);                                  \
    }
#define SKL_TEST_PARAM_I8(NAME, VAR)                                          \
    else if (strcmp(params[i].param, #NAME) == 0) {                            \
        VAR = (int8_t)atoi(params[i].value);                                   \
    }
#define SKL_TEST_PARAM_I32(NAME, VAR)                                          \
    else if (strcmp(params[i].param, #NAME) == 0) {                            \
        VAR = (int32_t)atoi(params[i].value);                                  \
    }
#define SKL_TEST_PARAM_F32(NAME, VAR)                                          \
    else if (strcmp(params[i].param, #NAME) == 0) {                            \
        VAR = (float)atof(params[i].value);                                    \
    }

#define SKL_TEST_PARAMS(PARAMS, NUM_PARAMS, ...)                                    \
    do {                                                                       \
        skl_test_param_t *params = (PARAMS);                                  \
        for (size_t i = 0; i < (NUM_PARAMS); ++i) {                            \
            if (0) {                                                           \
            }
            __VA_ARGS__ \
            else {                                                             \
                fprintf(stderr, "Unknown parameter: %s\n", params[i].param);   \
                return 1;                                                      \
            }                                                                  \
        }                                                                      \
    } while (0)


#define SKL_TEST_INIT_IMPL(CFG, TYPE, BUF, NUM, RAND)                                    \
    do {                                                                       \
        TYPE *buf = (BUF);                                                     \
        const size_t num = (NUM);                                              \
        switch (CFG.data) {                                        \
        case SKL_TEST_RANDOM:                                                  \
            for (size_t i = 0; i < num; ++i) {                                 \
                buf[i] = RAND;                        \
            }                                                                  \
            break;                                                             \
        case SKL_TEST_SEQ:                                                     \
            for (size_t i = 0; i < num; ++i) {                                 \
                buf[i] = (TYPE)(min + step * i);                               \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case SKL_TEST_STATIC:                                                  \
            for (size_t i = 0; i < num; ++i) {                                 \
                buf[i] = SKL_TEST_STATIC_DATA_NAME(BUF)[i % SKL_TEST_STATIC_DATA_LEN(BUF)]; \                                              \
            }                                                                  \
            break;                                                             \
        }
    } while (0)

#define SKL_TEST_INIT__Float16(CFG, BUF, NUM) \
    SKL_TEST_INIT_IMPL(CFG, _Float16, BUF, NUM, \
        ((float)rand() / (float)RAND_MAX) * (CFG.float_max - CFG.float_min) + CFG.float_min \
    )

#define SKL_TEST_INIT_float(CFG, BUF, NUM) \
    SKL_TEST_INIT_IMPL(CFG, float, BUF, NUM, \
        ((float)rand() / (float)RAND_MAX) * (CFG.float_max - CFG.float_min) + CFG.float_min \
    )

#define SKL_TEST_INIT_double(CFG, BUF, NUM) \
    SKL_TEST_INIT_IMPL(CFG, double, BUF, NUM, \
        ((double)rand() / (double)RAND_MAX) * (CFG.float_max - CFG.float_min) + CFG.float_min \
    )

#define SKL_TEST_INIT_int8_t(CFG, BUF, NUM) \
    SKL_TEST_INIT_IMPL(CFG, int8_t, BUF, NUM, \
        (int8_t)rand() % (CFG.int8_max - CFG.int8_min + 1) + CFG.int8_min \
    )

#define SKL_TEST_INIT_int32_t(CFG, BUF, NUM) \
    SKL_TEST_INIT_IMPL(CFG, int32_t, BUF, NUM, \
        (int32_t)rand() % (CFG.int32_max - CFG.int32_min + 1) + CFG.int32_min \
    )

#define SKL_TEST_BUFFER(NAME, TYPE, NUM) \
    *NAME = (TYPE *)SKL_TEST_MEMALIGN(SKL_TEST_ALIGNMENT, (NUM) * sizeof(TYPE)) \
    SKL_TEST_BUFFER_INIT_##TYPE(NAME, NUM)

/**
 * @file skl-test.c
 * 
 * Example implementation:
 * @code
 * 
 * typedef void (*gemm_test_func_t)(size_t, size_t, size_t, float, const float *, size_t, const float *, size_t, float, float *, size_t);
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
 *         { "FUNC", "skl_gemm_f32_f32_f32_zve32f_x390" }
 *     },
 *     .num_params = 6,
 * };
 * 
 * // The buffers to use for the test.
 * static float *A, *B, *C;
 * 
 * int SKL_TEST_CONFIG(skl_test_param_t *params, size_t num_params) {
 *     // Parse the configurable parameters.
 *     SKL_TEST_PARAMS(params, num_params,
 *         SKL_TEST_PARAM_SZ (M,        gemm_params.M),
 *         SKL_TEST_PARAM_SZ (N,        gemm_params.N),
 *         SKL_TEST_PARAM_SZ (K,        gemm_params.K),
 *         SKL_TEST_PARAM_F32(ALPHA,    gemm_params.ALPHA),
 *         SKL_TEST_PARAM_F32(BETA,     gemm_params.BETA)
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
 *     gemm_params.FUNC(gemm_params.M, gemm_params.N, gemm_params.K, gemm_params.ALPHA, A, gemm_params.RSA, B, gemm_params.RSB, gemm_params.BETA, C, gemm_params.RSC);
 *     return 0;
 * }
 * 
 * int SKL_TEST_VERIFY(void) {
 *     // Verify the results unless this is a performance test.
 *     return 0;
 * }
 * 
 * 
 * @endcode
 */

 /**
 * @brief Called by main() to configure the test.
 *
 * @param params The parameters to configure the test.
 * @param num_params The number of parameters.
 * @return 0 on success, non-zero on failure.
 * 
 * This function is called by main() to configure the test.
 * It should parse the parameters and set the test parameters accordingly.
 * It should also allocate and initialize the buffers used by the test.
 */
int SKL_TEST_CONFIG(skl_test_param_t *params, size_t num_params);

/**
 * @brief Called by main() to execute the test.
 *
 * @return 0 on success, non-zero on failure.
 * 
 * This function is called by main() to execute the test.
 * It should call the function(s) being tested.
 */
int SKL_TEST_EXECUTE(void);

/**
 * @brief Called by main() to verify the results of the test.
 *
 * @return 0 on success, non-zero on failure.
 * 
 * This function is called by main() to verify the results of the test.
 * It should compare the results against the expected results.
 */
int SKL_TEST_VERIFY(void);

int main(void) {
    int res = 0;

    res += SKL_TEST_CONFIG(SKL_TEST_PARAMS, SKL_TEST_NUM_PARAMS);
    res += SKL_TEST_EXECUTE();
    res += SKL_TEST_VERIFY();

    return res;
}
)