// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_pack_e8rc_e8rcprc_ref(size_t m, size_t n,
                                        const uint8_t *SKL_RESTRICT src,
                                        size_t rs, size_t cs, size_t m0,
                                        size_t n0, uint8_t *SKL_RESTRICT dst,
                                        size_t rs0, size_t cs0, size_t rs1,
                                        size_t cs1, uint8_t pad) {
  size_t m1 = (m + m0 - 1) / m0; // number of block-rows
  size_t n1 = (n + n0 - 1) / n0; // number of block-columns

  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0 * cs;
      uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;

      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
            dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0 * cs];
          } else {
            // Pad with zeros
            dst_block[ii0 * rs0 + jj0 * cs0] = pad;
          }
        }
      }
    }
  }
}
