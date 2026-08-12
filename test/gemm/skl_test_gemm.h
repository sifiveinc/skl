// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @file skl_test_gemm.h
 * @brief Functions shared among the GEMM test harnesses.
 */

#pragma once

#include "skl-test-driver.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__riscv_min_xsfmm_te)
#define SKL_XSFMM_TE ((size_t)__riscv_min_xsfmm_te)
#endif

#ifdef __riscv_xsfmmbase
/**
 * @brief Get the effective tile edge length.
 */
static inline size_t skl_get_ete_xsfmmbase(void) {
  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e8, w1" : "=r"(ete) : : "vtype", "vl");
  return ete;
}
#endif

/**
 * @brief Check for clobbered elements in a packed matrix
 *
 * @param t - Test context.
 * @param element_size - Size of the elements of X and Y in bytes.
 * @param len - Number of elements in the arrays for X and Y.
 * @param m0 - Number of rows in each block of X and Y.
 * @param n0 - Number of columns in each block of X and Y.
 * @param m1 - Number of block-rows in X and Y.
 * @param n1 - Number of block-columns in X and Y.
 * @param x - Pointer to matrix X.
 * @param y - Pointer to matrix Y.
 * @param rs0 - Row stride within each block of X and Y in elements.
 * @param cs0 - Column stride within each block of X and Y in elements.
 * @param rs1 - Row stride between blocks of X and Y in elements.
 * @param cs1 - Column stride between blocks of X and Y in elements.
 *
 * X and Y are packed matrices stored in arrays with len elements of size
 * element_size bytes. The elements of the matrices are indexed by:
 *   i1 * rs1 + j1 * cs1 + i0 * rs0 + j0 * cs0,
 * for 0 <= i1 < m1, 0 <= j1 < n1, 0 <= i0 < m0, 0 <= j0 < n0. This function
 * checks that the arrays pointed to by X and Y are identical outside of the
 * above indices and updates the verify_status of t.
 */
static inline void skl_test_check_matrix_clobbered_rcprc(
    skl_test_t *t, size_t element_size, size_t len, size_t m0, size_t n0,
    size_t m1, size_t n1, void *x, void *y, size_t rs0, size_t cs0, size_t rs1,
    size_t cs1) {
  uint8_t *x_u8 = (uint8_t *)x;
  uint8_t *y_u8 = (uint8_t *)y;
  // When m0 == 1, rs0 can be arbitrary. We adjust its value in this case to
  // reflect the length of a row. cs0, rs1, and cs1 are adjusted similarly.
  size_t rs0_adj = m0 == 1 ? (n0 - 1) * cs0 + 1 : rs0;
  size_t cs0_adj = n0 == 1 ? (m0 - 1) * rs0 + 1 : cs0;
  size_t block_min_len = (m0 - 1) * rs0_adj + (n0 - 1) * cs0_adj + 1;
  size_t rs1_adj = m1 == 1 ? (n1 - 1) * cs1 + block_min_len : rs1;
  size_t cs1_adj = n1 == 1 ? (m1 - 1) * rs1 + block_min_len : cs1;
  // Each block of X and Y lies in an ambient m0_amb x n0_amb row- or
  // column-major matrix with row and column strides rs0_amb and cs0_amb.
  // Let bs1 be the block stride of X and Y, i.e. the distance from one block to
  // the next in memory, in elements; bs1 varies from block to block.
  // Lastly, set layout0_row_major to true if blocks of X and Y are row major
  // and false otherwise. Similarly, set layout1_row_major to true if X and Y
  // are block-row-major and false otherwise.
  size_t m0_amb = 0;
  size_t n0_amb = 0;
  size_t rs0_amb = 0;
  size_t cs0_amb = 0;
  size_t bs1 = 0;
  bool layout0_row_major = rs0_adj >= cs0_adj;
  bool layout1_row_major = rs1_adj >= cs1_adj;
  if (layout0_row_major) {
    rs0_amb = rs0_adj;
    cs0_amb = 1;
    m0_amb = m0;
    n0_amb = rs0_amb;
  } else {
    rs0_amb = 1;
    cs0_amb = cs0_adj;
    m0_amb = cs0_amb;
    n0_amb = n0;
  }

  // Check each block.
  for (size_t i1 = 0; i1 < m1; ++i1) {
    for (size_t j1 = 0; j1 < n1; ++j1) {
      // Check the first block_min_len elements.
      for (size_t i0 = 0; i0 < m0_amb; ++i0) {
        for (size_t j0 = 0; j0 < n0_amb; ++j0) {
          if (layout0_row_major && j0 % cs0_adj == 0 && j0 < n0 * cs0_adj) {
            continue;
          }
          if (!layout0_row_major && i0 % rs0_adj == 0 && i0 < m0 * rs0_adj) {
            continue;
          }
          if (i0 * rs0_amb + j0 * cs0_amb >= block_min_len) {
            continue;
          }

          size_t idx =
              i1 * rs1_adj + j1 * cs1_adj + i0 * rs0_amb + j0 * cs0_amb;
          int64_t element0 = 0;
          int64_t element1 = 0;
          memcpy(&element0, &(x_u8[element_size * idx]), element_size);
          memcpy(&element1, &(y_u8[element_size * idx]), element_size);
          if (element0 != element1) {
            SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                         "element [%zu, %zu, %zu, %zu] clobbered\n", i1, j1, i0,
                         j0);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }

      // Check until the next block.
      if (layout1_row_major) {
        bs1 = j1 < n1 - 1 ? cs1_adj : rs1_adj - (n1 - 1) * cs1_adj;
      } else {
        bs1 = i1 < m1 - 1 ? rs1_adj : cs1_adj - (m1 - 1) * rs1_adj;
      }
      if (i1 == m1 - 1 && j1 == n1 - 1) {
        bs1 = block_min_len;
      }
      for (size_t i = block_min_len; i < bs1; ++i) {
        size_t idx = i1 * rs1_adj + j1 * cs1_adj + i;
        int64_t element0 = 0;
        int64_t element1 = 0;
        memcpy(&element0, &(x_u8[element_size * idx]), element_size);
        memcpy(&element1, &(y_u8[element_size * idx]), element_size);
        if (element0 != element1) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "element [%zu, %zu, %zu] clobbered\n", i1, j1, i);
          t->status.verify_status = SKL_TEST_FAIL;
          return;
        }
      }
    }
  }

  // Check until the end of the array.
  size_t min_len = m1 == 0 || n1 == 0 ? 0
                                      : (m1 - 1) * rs1_adj +
                                            (n1 - 1) * cs1_adj + block_min_len;
  for (size_t idx = min_len; idx < len; ++idx) {
    int64_t element0 = 0;
    int64_t element1 = 0;
    memcpy(&element0, &(x_u8[element_size * idx]), element_size);
    memcpy(&element1, &(y_u8[element_size * idx]), element_size);
    if (element0 != element1) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "element [%zu] clobbered\n", idx);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

/**
 * @brief Print out the matrix dimensions and strides for a packed GEMM test.
 *
 * @param t - Test context.
 * @param m0 - Number of rows in each block of matrices A_pack and C_pack.
 * @param n0 - Number of columns in each block of matrices B_pack and C_pack.
 * @param k0 - Number of columns in each block of A_pack and rows in each block
 *             of B_pack.
 * @param m1 - Number of rows in A_pack and C_pack as block matrices.
 * @param n1 - Number of columns in B_pack and C_pack as block matrices.
 * @param k1 - Number of columns in A_pack and rows in B_pack as block matrices.
 * @param rsa0 - Row stride within each block of A_pack in elements.
 * @param csa0 - Column stride within each block of A_pack in elements.
 * @param rsa1 - Row stride between blocks of A_pack in elements.
 * @param csa1 - Column stride between blocks of A_pack in elements.
 * @param rsb0 - Row stride within each block of B_pack in elements.
 * @param csb0 - Column stride within each block of B_pack in elements.
 * @param rsb1 - Row stride between blocks of B_pack in elements.
 * @param csb1 - Column stride between blocks of B_pack in elements.
 * @param rsc0 - Row stride within each block of C_pack in elements.
 * @param csc0 - Column stride within each block of C_pack in elements.
 * @param rsc1 - Row stride between blocks of C_pack in elements.
 * @param csc1 - Column stride between blocks of C_pack in elements.
 */
static inline void gemm_rcprc_rcprc_rcprc_report_matrix_params(
    skl_test_t *t, size_t m0, size_t n0, size_t k0, size_t m1, size_t n1,
    size_t k1, size_t rsa0, size_t csa0, size_t rsa1, size_t csa1, size_t rsb0,
    size_t csb0, size_t rsb1, size_t csb1, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1) {

  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "M0: %zu, N0: %zu, K0: %zu\n", m0, n0, k0);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "M1: %zu, N1: %zu, K1: %zu\n", m1, n1, k1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO,
               "RSA0: %zu, CSA0: %zu, RSA1: %zu, CSA1: %zu\n", rsa0, csa0, rsa1,
               csa1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO,
               "RSB0: %zu, CSB0: %zu, RSB1: %zu, CSB1: %zu\n", rsb0, csb0, rsb1,
               csb1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO,
               "RSC0: %zu, CSC0: %zu, RSC1: %zu, CSC1: %zu\n", rsc0, csc0, rsc1,
               csc1);
}

/**
 * @brief Print out performance information for a packed GEMM test.
 *
 * @param t - Test context.
 * @param warmup - Pointer to the test harness's warmup function.
 * @param maccs - Number of multiply-accumulate operations performed.
 */
static inline void
gemm_rcprc_rcprc_rcprc_report_perf(skl_test_t *t, void (*warmup)(skl_test_t *),
                                   size_t maccs) {
  uint64_t cycles = t->counters.cycles;
  float mpc = (float)maccs / (float)cycles;

  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Warmup: %s\n", warmup ? "yes" : "no");
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Cycles: %" PRIu64 "\n", cycles);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Instructions: %" PRIu64 "\n",
               t->counters.instret);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "MACCs/Cycle: %f\n", mpc);
}
