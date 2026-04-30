// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_exp_f16_ref(_Float16 *out, const _Float16 *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = (_Float16)expf(in[i]);
  }
}
