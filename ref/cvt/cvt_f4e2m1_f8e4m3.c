// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <stdint.h>

// NOLINTBEGIN(*)
// TODO(fix linting errors)
SKL_FUNC float skl_cvt_f4e2m1_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 3)
  uint32_t sign = (in & 0x08) << 28; // Shift to FP32 sign position

  // E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bits
  uint32_t exponent = (in >> 1) & 0x03; // Bits 2-1
  uint32_t mantissa = in & 0x01;        // Bit 0

  if (exponent == 0) {
    // Subnormal E2M1: value = mantissa / 2^1
    if (mantissa != 0) {
      int lz = __builtin_clz(mantissa << 31);
      uint32_t msk = ~(uint32_t)0 << (1 - (lz + 1));
      uint32_t man = (mantissa & ~msk) << (23 - 1 + lz + 1);
      uint32_t exp = (1 - (lz + 1) - 1 + 127) << 23;
      result = sign | exp | man;
    } else {
      // Zero
      result = sign;
    }
  } else {
    // Normal numbers
    // Bias: E2M1 uses 1, FP32 uses 127
    uint32_t unbiased_exp = exponent - 1 + 127;
    result = sign | (unbiased_exp << 23) | (mantissa << 22);
  }

  return skl_u32_as_float(result);
}
// NOLINTEND(*)
