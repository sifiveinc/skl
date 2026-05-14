// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_logistic_1u_bf16_ref(__bf16 *out, const __bf16 *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const float x = in[i];
    const float a = copysignf (x, -1.f);
    const float e = expf (a);
    const float d = e + 1.f;
    const float q = (x < 0.f) ? e / d : 1.f / d;
    out[i] = (__bf16)q;
  }
}
