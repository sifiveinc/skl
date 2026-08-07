// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

/**
 * Exponential function approximation on a vector of f32 floating-point values
 * with a 1-ULP error bound in [-inf; 0].
 *
 * @note NaNs are not propagated.
 */
SKL_FUNC void skl_sigmoid_f32_ref(float *out, float beta, const float *x,
                                  const float *y, const float *up, float delta,
                                  size_t n) {
  if (!x)
    return;

  for (size_t i = 0; i < n; i++) {
    double a = x[i];
    /* Sigmoid */
    double b = beta * a;
    double e = exp(copysign(b, -1));
    double o = 1 + e;
    if (b >= 0)
      o = 1 / o;
    else
      o = e / o;

    /* Linear unit */
    if (y)
      o *= y[i];
    /* Gate */
    if (up)
      o *= (up[i] + delta);

    out[i] = (float)o;
  }
}

SKL_FUNC void skl_logistic_f32_ref(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_ref(out, 1.f, x, NULL, NULL, 0.f, n);
}

SKL_FUNC void skl_silu_f32_ref(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_ref(out, 1.f, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swish_f32_ref(float *out, float beta, const float *x,
                                size_t n) {
  skl_sigmoid_f32_ref(out, beta, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_glu_f32_ref(float *out, const float *x, const float *y,
                              size_t n) {
  skl_sigmoid_f32_ref(out, 1.f, y, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swiglu_f32_ref(float *out, const float *gate, const float *up,
                                 float delta, size_t n) {
  skl_sigmoid_f32_ref(out, 1.f, gate, gate, up, delta, n);
}
