// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include "skl-test-driver.h"

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

/**
 * @brief Called by main() to report benchmark-specific metrics.
 *
 * @param cycles The number of cycles.
 * @param insts The number of instructions.
 * @return 0 on success, non-zero on failure.
 *
 */
int SKL_TEST_REPORT(uint64_t cycles, uint64_t insts);

/**
 * @brief Called by main() to finish the test.
 *
 * @return 0 on success, non-zero on failure.
 *
 * This function is called by main() to finish the test.
 * It can be used to deallocate the buffers used by the test.
 */
int SKL_TEST_FINISH(void);

int main(void) {
  int res = 0;

  res += SKL_TEST_CONFIG(SKL_TEST_PARAMS, SKL_TEST_NUM_PARAMS);

  if (skl_test_config.warm_cache) {
    res += SKL_TEST_EXECUTE();
  }

  uint64_t c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();
  res += SKL_TEST_EXECUTE();
  uint64_t c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();
  uint64_t cycles = c1 - c0;
  uint64_t insts = i1 - i0;

  SKL_TEST_RESULT("TEST", "%s", skl_test_config.name);
  SKL_TEST_RESULT("CYCLES", "%lu", cycles);
  SKL_TEST_RESULT("INSTS", "%lu", insts);
  res += SKL_TEST_REPORT(cycles, insts);

  if (skl_test_config.verify) {
    res += SKL_TEST_VERIFY();
  }

  res += SKL_TEST_FINISH();

  return res;
}