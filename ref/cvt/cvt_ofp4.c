// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include "skl-ref.h"
#include <stddef.h>
#include <stdint.h>

/** @brief Reinterprets an unsigned 32-bit integer as an IEEE FP32. */
SKL_FUNC_PRIVATE float skl_u32_as_float(uint32_t x) {
  float y;
  __builtin_memcpy(&y, &x, sizeof(float));
  return y;
}

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

SKL_FUNC_PRIVATE void cvt_ofp4x2_f8e4m3(uint8_t in, uint8_t *out0,
                                        uint8_t *out1) {
  // Extract lower and upper 4-bit values from the packed input byte
  uint8_t in_0 = in & 0xFU;
  uint8_t in_1 = in >> 4U;
  *out0 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_0), false);
  *out1 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_1), false);
}

SKL_FUNC void skl_cvt_f4e2m1_f8e4m3_ref(uint8_t *pDst, const uint8_t *pSrc,
                                        size_t n) {
  for (size_t i = 0; i < n / 2 * 2; i += 2) {
    cvt_ofp4x2_f8e4m3(pSrc[i / 2], pDst + i, pDst + i + 1);
  }
  if (n % 2 == 1) {
    uint8_t tmp;
    cvt_ofp4x2_f8e4m3(pSrc[n / 2], pDst + n - 1, &tmp);
  }
}
