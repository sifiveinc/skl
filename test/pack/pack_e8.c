// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the pack_e8 test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/skl_pack_e8_zve32x.c).
 */

#include "pack_e8.h"
#include "skl-test-driver.h"
#include "skl.h" // NOLINT(misc-include-cleaner)
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO(hchen): Move the reference function to ref/
// Reference implementation for pack
static void skl_pack_e8rc_e8rcbrc_ref(
    size_t m,             // Num. rows in input matrix
    size_t n,             // Num. columns in input matrix
    const uint8_t *src,   // Input matrix
    size_t rs,            // Row stride of input matrix
    size_t cs,            // Column stride of input matrix
    size_t m0,            // Num. rows in a block of the input matrix
    size_t n0,            // Num. columns in a block of the input matrix
    uint8_t *dst,         // Output packed matrix
    size_t rs0,           // Row stride within a block of the output matrix
    size_t cs0,           // Column stride within a block of the output matrix
    size_t rs1,           // Row stride between blocks of the output matrix
    size_t cs1,           // Column stride between blocks of the output matrix
    uint8_t padding_value // Value to use for padding
) {
  size_t m1 = (m + m0 - 1) / m0; // Num. row blocks
  size_t n1 = (n + n0 - 1) / n0; // Num. column blocks

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
            dst_block[ii0 * rs0 + jj0 * cs0] = padding_value;
          }
        }
      }
    }
  }
}

void pack_e8_init(skl_test_t *t) {
  pack_e8_t *h = (pack_e8_t *)t->harness;

  // Calculate source buffer size
  // For row-major: m rows each of stride rs
  // For column-major: n columns each of stride cs
  // General: max(m * rs, n * cs) to ensure we cover the whole matrix
  size_t src_len_rowwise = h->m * h->rs;
  size_t src_len_colwise = h->n * h->cs;
  h->src.len =
      src_len_rowwise > src_len_colwise ? src_len_rowwise : src_len_colwise;

  // Calculate destination buffer size
  size_t m1 = (h->m + h->m0 - 1) / h->m0;
  size_t n1 = (h->n + h->n0 - 1) / h->n0;
  // Size needs to accommodate all blocks: m1 blocks vertically, n1 blocks
  // horizontally
  h->ctx.dst_size = (m1 > 0 ? (m1 - 1) * h->rs1 : 0) +
                    (n1 > 0 ? (n1 - 1) * h->cs1 : 0) + h->m0 * h->rs0 +
                    h->n0 * h->cs0;
  h->dst.len = h->ctx.dst_size;

  // For row-major (cs=1): rs >= n * cs (each row must fit n elements)
  // For column-major (rs=1): cs >= m * rs (each column must fit m elements)
  // General case: the stride must accommodate the dimension

  SKL_TEST_REQUIRE(t, init_status,
                   h->rs >= h->n * h->cs || h->cs >= h->m * h->rs);
  SKL_TEST_REQUIRE(t, init_status,
                   h->rs0 >= h->n0 * h->cs0 || h->cs0 >= h->m0 * h->rs0);

  if (h->rs0 >= h->n0) {
    SKL_TEST_REQUIRE(
        t, init_status,
        (h->rs0 * h->m0 * n1 <= h->rs1 &&
         h->rs0 * h->m0 <= h->cs1) // blocks in row-major
            || (h->rs0 * h->m0 <= h->rs1 &&
                h->rs0 * h->m0 * m1 <= h->cs1) // blocks in column-major
    );
  }
  if (h->cs0 >= h->m0) {
    SKL_TEST_REQUIRE(
        t, init_status,
        (h->cs0 * h->n0 * n1 <= h->rs1 &&
         h->cs0 * h->n0 <= h->cs1) // blocks in row-major
            || (h->cs0 * h->n0 <= h->rs1 &&
                h->cs0 * h->n0 * m1 <= h->cs1) // blocks in column-major
    );
  }

  if (t->status.init_status != SKL_TEST_PASS)
    return;

  SKL_TEST_BUF_CREATE(t, uint8_t, &h->src);
  SKL_TEST_BUF_CREATE(t, uint8_t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    h->ctx.ref_dst = malloc(h->dst.len * sizeof(uint8_t));

    // Copy original `dst` contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_dst, h->dst.data, h->dst.len * sizeof(uint8_t));
  }
}

void pack_e8_verify(skl_test_t *t) {
  pack_e8_t *h = (pack_e8_t *)t->harness;

  const uint8_t *dst = h->dst.data;
  uint8_t *ref_dst = h->ctx.ref_dst;

  // Compute reference value
  skl_pack_e8rc_e8rcbrc_ref(h->m, h->n, h->src.data, h->rs, h->cs, h->m0, h->n0,
                            ref_dst, h->rs0, h->cs0, h->rs1, h->cs1,
                            h->padding_value);

  // Verify result
  for (size_t i = 0; i < h->dst.len; ++i) {
    if (dst[i] != ref_dst[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "position [%zu]: %hhu != ref %hhu\n",
                   i, dst[i], ref_dst[i]);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

void pack_e8_param_report(skl_test_t *t) {
  pack_e8_t *h = (pack_e8_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  size_t m1 = (h->m + h->m0 - 1) / h->m0;
  size_t n1 = (h->n + h->n0 - 1) / h->n0;
  INFO("M: %zd, N: %zd\n", h->m, h->n);
  INFO("M0: %zd, N0: %zd\n", h->m0, h->n0);
  INFO("M1: %zd, N1: %zd\n", m1, n1);
  INFO("RS: %zd, CS: %zd\n", h->rs, h->cs);
  INFO("RS0: %zd, CS0: %zd\n", h->rs0, h->cs0);
  INFO("RS1: %zd, CS1: %zd\n", h->rs1, h->cs1);
  if (h->rs == 1) {
    if (h->rs0 == 1) {
      INFO("%s", "=> COPY case!\n");
    } else if (h->cs0 == 1) {
      INFO("%s", "=> TRANSPOSE case!\n");
    } else {
      INFO("%s", "=> GENERAL case!\n");
    }
  } else if (h->cs == 1) {
    if (h->rs0 == 1) {
      INFO("%s", "=> TRANSPOSE case!\n");
    } else if (h->cs0 == 1) {
      INFO("%s", "=> COPY case!\n");
    } else {
      INFO("%s", "=> GENERAL case!\n");
    }
  }
#undef INFO
}

void pack_e8_bench_report(skl_test_t *t) {

  pack_e8_param_report(t);
  pack_e8_t *h = (pack_e8_t *)t->harness;
#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
#undef INFO
}

void pack_e8_cleanup(skl_test_t *t) {
  pack_e8_t *h = (pack_e8_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->src);
  SKL_TEST_BUF_FREE(t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    free(h->ctx.ref_dst);
  }
}
