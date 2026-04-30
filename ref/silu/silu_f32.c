// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_silu_52u_f32_ref(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float x = in[i];
    if (x <= 0.0f) {
      float ex = expf(x);
      out[i] = x * ex / (1.0f + ex);
    } else {
      /* For x > 0, compute silu(x) = x * (1 - logistic(-x)) */
      float ex = expf(-x);
      out[i] = x * (1.0f - ex / (1.0f + ex));
    }
  }
}
