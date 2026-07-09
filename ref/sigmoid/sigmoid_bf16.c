// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_bf16_ref(__bf16 *out, __bf16 beta, const __bf16 *x,
                                   const __bf16 *y, const __bf16 *up,
                                   __bf16 delta, size_t n) {
  if (!x)
    return;

  for (size_t i = 0; i < n; i++) {
    float o;
    float a = x[i];
    /* Sigmoid */
    float b = beta * a;
    if (b >= 0)
      o = 1 / (1 + expf(-b));
    else {
      float e = expf(b);
      o = e / (1 + e);
    }
    /* Linear unit */
    if (y)
      o *= y[i];
    /* Gate */
    if (up) {
      o *= (up[i] + delta);
    }
    out[i] = (__bf16)o;
  }
}

SKL_FUNC void skl_logistic_bf16_ref(__bf16 *out, const __bf16 *x, size_t n) {
  skl_sigmoid_bf16_ref(out, (__bf16)1, x, NULL, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_silu_bf16_ref(__bf16 *out, const __bf16 *x, size_t n) {
  skl_sigmoid_bf16_ref(out, (__bf16)1, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swish_bf16_ref(__bf16 *out, __bf16 beta, const __bf16 *x,
                                 size_t n) {
  skl_sigmoid_bf16_ref(out, beta, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_glu_bf16_ref(__bf16 *out, const __bf16 *x, const __bf16 *y,
                               size_t n) {
  skl_sigmoid_bf16_ref(out, (__bf16)1, y, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swiglu_bf16_ref(__bf16 *out, const __bf16 *gate,
                                  const __bf16 *up, __bf16 delta, size_t n) {
  skl_sigmoid_bf16_ref(out, (__bf16)1, gate, gate, up, delta, n);
}
