// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Convert a single OFP4 E2M1 value to IEEE FP32.
 *
 * @param in - Input value. The lowest 4 bits contain the OFP4 E2M1 value.
 * @return The converted IEEE FP32 value.
 */
float skl_cvt_f4e2m1_f32(uint8_t in);

/** @brief Convert packed OFP4 E2M1 values to OFP8 E4M3 format (reference
 * implementation).
 *
 * @param pDst - Output array for converted OFP8 E4M3 values.
 * @param pSrc - Input array containing packed OFP4 E2M1 values (2 elements per
 * byte).
 * @param n - Number of elements to convert.
 */
void skl_cvt_f4e2m1_f8e4m3_ref(uint8_t *pDst, const uint8_t *pSrc, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
