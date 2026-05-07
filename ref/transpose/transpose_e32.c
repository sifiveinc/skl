// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_transpose_e32_ref(size_t m, size_t n,
                                    const uint32_t *SKL_RESTRICT a, size_t rsa,
                                    uint32_t *SKL_RESTRICT at, size_t rsat) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      at[j * rsat + i] = a[i * rsa + j];
    }
  }
}
