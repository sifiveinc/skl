// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <stdint.h>

SKL_FUNC float skl_cvt_f4e2m1_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 3)
  uint32_t sign = ((uint32_t)in & 0x08U) << 28U; // Shift to FP32 sign position

  // E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bits
  uint32_t exponent = ((uint32_t)in >> 1U) & 0x03U; // Bits 2-1
  uint32_t mantissa = (uint32_t)in & 0x01U;         // Bit 0

  if (exponent == 0U) {
    // Subnormal E2M1: value = mantissa / 2^1
    if (mantissa != 0U) {
      int lz = __builtin_clz(mantissa << 31U);
      uint32_t msk = ~0U << (uint32_t)(1 - (lz + 1));
      uint32_t man = (mantissa & ~msk) << (uint32_t)(23 - 1 + lz + 1);
      uint32_t exp = (uint32_t)(1 - (lz + 1) - 1 + 127) << 23U;
      result = sign | exp | man;
    } else {
      // Zero
      result = sign;
    }
  } else {
    // Normal numbers
    // Bias: E2M1 uses 1, FP32 uses 127
    uint32_t unbiased_exp = exponent - 1U + 127U;
    result = sign | (unbiased_exp << 23U) | (mantissa << 22U);
  }

  return skl_u32_as_float(result);
}
