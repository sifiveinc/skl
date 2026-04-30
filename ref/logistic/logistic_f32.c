// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_logistic_3u_f32_ref(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float x = in[i];
    if (!isnan(x)) {
      if (x <= 0.0f) {
        float ex = expf(x);
        out[i] = ex / (1.0f + ex);
      } else {
        /* For x > 0, compute logistic(x) = 1 - logistic(-x) */
        float ex = expf(-x);
        out[i] = 1.0f - ex / (1.0f + ex);
      }
    } else {
      out[i] = x + x; // propagate NaN
    }
  }
}
