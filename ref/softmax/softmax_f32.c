// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f32_ref(float *pDst, const float *pSrc, float beta,
                                  size_t n) {
  if (n < 1) {
    return;
  }

  float max = pSrc[0];
  for (size_t i = 1; i < n; i++) {
    max = fmaxf(pSrc[i], max);
  }

  float sum = 0;
  for (size_t i = 0; i < n; i++) {
    pDst[i] = expf(beta * (pSrc[i] - max));
    sum += pDst[i];
  }

  float recip_sum = 1.0f / sum;
  for (size_t i = 0; i < n; i++) {
    pDst[i] *= recip_sum;
  }
}

SKL_FUNC void skl_softmax_2d_f32_ref(float *s, const size_t rss, const float *a,
                                     const size_t rsa, const float beta,
                                     const size_t m, const size_t n) {
  for (size_t i = 0; i < m; ++i) {
    skl_softmax_f32_ref(s + i * rss, a + i * rsa, beta, n);
  }
}
