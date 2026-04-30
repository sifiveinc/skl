// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f16_ref(_Float16 *pDst, const _Float16 *pSrc,
                                  _Float16 beta, size_t n) {
  if (n < 1) {
    return;
  }

  _Float16 max = pSrc[0];
  for (size_t i = 1; i < n; i++) {
    max = (_Float16)fmaxf(pSrc[i], max);
  }

  float sum = 0;
  for (size_t i = 0; i < n; i++) {
    float e = expf(beta * (pSrc[i] - max));
    pDst[i] = (_Float16)e;
    sum += e;
  }

  float recip_sum = 1.0f / sum;
  for (size_t i = 0; i < n; i++) {
    pDst[i] = (_Float16)(pDst[i] * recip_sum);
  }
}
