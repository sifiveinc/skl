// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdbool.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Function to convert OFP8 E4M3 to IEEE FP32.
 *
 * @param in - An OFP8 8-bit floating point number in E4M3 format, type-punned
 * as an 8-bit unsigned integer.
 */
float skl_cvt_f8e4m3_f32(uint8_t in);

/** @brief Function to convert OFP8 E5M2 to IEEE FP32.
 *
 * @param in - An OFP8 8-bit floating point number in E5M2 format, type-punned
 * as an 8-bit unsigned integer.
 */
float skl_cvt_f8e5m2_f32(uint8_t in);

/** @brief Function to convert IEEE FP32 to OFP8 E4M3.
 *
 * @param in - An OFP8 8-bit floating point number in E4M3 format, type-punned
 * as an 8-bit unsigned integer.
 * @param is_sat - Whether to saturate the output to the maximum representable
 * value of the same sign when the input is out of range.
 */
uint8_t skl_cvt_f32_f8e4m3(float in, bool is_sat);

/** @brief Function to convert IEEE FP32 to OFP8 E5M2.
 *
 * @param in - An OFP8 8-bit floating point number in E5M2 format, type-punned
 * as an 8-bit unsigned integer.
 * @param is_sat - Whether to saturate the output to the maximum representable
 * value of the same sign when the input is out of range.
 */
uint8_t skl_cvt_f32_f8e5m2(float in, bool is_sat);

#if defined(__cplusplus)
} // extern "C"
#endif
