// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_ref(__bf16 *pDst, const __bf16 *pSrc,
                                   __bf16 beta, size_t n) {
  if (n < 1) {
    return;
  }

  float max = pSrc[0];
  for (size_t i = 1; i < n; i++) {
    max = fmaxf(pSrc[i], max);
  }

  float sum = 0;
  for (size_t i = 0; i < n; i++) {
    float e = expf(beta * (pSrc[i] - max));
    pDst[i] = (__bf16)e;
    sum += e;
  }

  float recip_sum = 1.0f / sum;
  for (size_t i = 0; i < n; i++) {
    pDst[i] = (__bf16)(pDst[i] * recip_sum);
  }
}
