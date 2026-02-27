// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

skl_test_config_t skl_test_config = {
    .name = "Unknown SKL test",
    .memalign = aligned_alloc,
    .free = free,
    .alignment = 64,
    .warm_cache = true,
    .verify = true,
    .data = SKL_TEST_RANDOM,
    .float_min = -10.0f,
    .float_max = 10.0f,
    .int32_min = -100,
    .int32_max = 100,
    .int8_min = -128,
    .int8_max = 127,
};

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
int skl_test_init(skl_test_param_t *params, size_t num_params);

/**
 * @brief Called by main() to execute the test.
 *
 * @return 0 on success, non-zero on failure.
 *
 * This function is called by main() to execute the test.
 * It should call the function(s) being tested.
 */
int skl_test_execute(void);

/**
 * @brief Called by main() to verify the results of the test.
 *
 * @return 0 on success, non-zero on failure.
 *
 * This function is called by main() to verify the results of the test.
 * It should compare the results against the expected results.
 */
int skl_test_verify(void);

/**
 * @brief Called by main() to report benchmark-specific metrics.
 *
 * @param cycles The number of cycles.
 * @param insts The number of instructions.
 * @return 0 on success, non-zero on failure.
 *
 */
int skl_test_report(uint64_t cycles, uint64_t insts);

/**
 * @brief Called by main() to finish the test.
 *
 * @return 0 on success, non-zero on failure.
 *
 * This function is called by main() to finish the test.
 * It can be used to deallocate the buffers used by the test.
 */
int skl_test_finish(void);

#if __riscv_xlen == 32
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint32_t lo;                                                               \
    uint32_t hi0, hi1;                                                         \
    /* guard against overflow between reads of lo/hi counter halves */         \
    do {                                                                       \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi0));                      \
      __asm__ volatile("rd" #COUNTER " %0" : "=r"(lo));                        \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi1));                      \
    } while (hi0 != hi1);                                                      \
    return (uint64_t)lo + ((uint64_t)hi1 << 32);                               \
  }
#elif __riscv_xlen == 64
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint64_t res;                                                              \
    __asm__ volatile("rd" #COUNTER " %0" : "=r"(res));                         \
    return res;                                                                \
  }
#endif

RISCV_READ_COUNTER_FUNC(cycle)   // riscv_read_mcycle()
RISCV_READ_COUNTER_FUNC(instret) // riscv_read_minstret()

int main(void) {
  int res = 0;

  res += skl_test_init(SKL_TEST_PARAMS, SKL_TEST_NUM_PARAMS);

  if (skl_test_config.warm_cache) {
    res += skl_test_execute();
  }

  uint64_t c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();
  res += skl_test_execute();
  uint64_t c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();
  uint64_t cycles = c1 - c0;
  uint64_t insts = i1 - i0;

  SKL_TEST_RESULT("TEST", "%s", skl_test_config.name);
  SKL_TEST_RESULT("CYCLES", "%lu", cycles);
  SKL_TEST_RESULT("INSTS", "%lu", insts);
  res += skl_test_report(cycles, insts);

  if (skl_test_config.verify) {
    res += skl_test_verify();
  }

  res += skl_test_finish();

  return res;
}