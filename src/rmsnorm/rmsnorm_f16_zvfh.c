// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_rmsnorm_f16_zvfh(const _Float16 *pDst, const _Float16 *pSrc,
                                   const _Float16 *pWeight, _Float16 epsilon,
                                   size_t n) {
  // Hello world - placeholder implementation
  // TODO: Implement RMS normalization using RISC-V vector intrinsics
  (void)pDst;
  (void)pSrc;
  (void)pWeight;
  (void)epsilon;
  (void)n;
}
