// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include "skl-common.h"
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference implementation of FP16 RMS normalization.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param pWeight - Array of weight elements (optional, can be NULL).
 * @param epsilon - Small value to avoid division by zero.
 * @param n - Number of elements to process.
 *
 * Computes: pDst[i] = pSrc[i] / rms * pWeight[i]
 * where rms = sqrt(sum(pSrc[i]^2) / n + epsilon)
 */
SKL_FUNC void skl_rmsnorm_f16_ref(const _Float16 *pDst, const _Float16 *pSrc,
                                  const _Float16 *pWeight, _Float16 epsilon,
                                  size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
