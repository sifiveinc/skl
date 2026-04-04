// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Function to convert OCP FP4 E2M1 to IEEE FP32.
 *
 * @param in - The lowest 4 bits of `in` contain the OCP FP4 4-bit floating
 * point number in E2M1 format.
 */
float skl_cvt_f4e2m1_f32(uint8_t in);

#if defined(__cplusplus)
} // extern "C"
#endif
