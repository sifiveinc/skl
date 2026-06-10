// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_rmsnorm_f32_ref(float *pDst, const float *pSrc,
                                  const float *pWeight, size_t rsc,
                                  float epsilon, size_t n) {
  // TODO: Implement reference RMS normalization
  size_t row_cnt = n/rsc;
  for (size_t r = 0; r < row_cnt; r++) {
    const float *cur_src = pSrc + (r * rsc);
    float *cur_dst = pDst + (r * rsc);

    float sum_sq = 0.0f;

    for (size_t i = 0; i < rsc; i++) {
      sum_sq += cur_src[i] * cur_src[i];
    }

    float mean_sq = sum_sq / (float)rsc;
    float inv_rms = 1.0f / sqrtf(mean_sq + epsilon);

    for (size_t i = 0; i < rsc; i++) {
      cur_dst[i] = cur_src[i] * inv_rms * pWeight[i];
    }
  }
}
