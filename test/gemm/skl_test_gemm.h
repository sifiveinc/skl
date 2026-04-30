// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

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
 * @brief Check packed matrix dimensions and strides
 *
 * @param t - Test context.
 * @param m0 - Number of rows in each block of the matrix.
 * @param n0 - Number of columns in each block of the matrix.
 * @param m1 - Number of block-rows in the matrix.
 * @param n1 - Number of block-columns in the matrix.
 * @param rs0 - Row stride within each block of the matrix in elements.
 * @param cs0 - Column stride within each block of the matrix in elements.
 * @param rs1 - Row stride between blocks of the matrix in elements.
 * @param cs1 - Column stride between blocks of the matrix in elements.
 *
 * This function performs some basic checks on the dimensions and strides of a
 * packed matrix and updates the init_status of t.
 */
static inline void skl_test_check_matrix_params_rcprc(skl_test_t *t, size_t m0,
                                                      size_t n0, size_t m1,
                                                      size_t n1, size_t rs0,
                                                      size_t cs0, size_t rs1,
                                                      size_t cs1) {
  SKL_TEST_REQUIRE(t, init_status, m0 > 0);
  SKL_TEST_REQUIRE(t, init_status, n0 > 0);

  if (m0 > 1) {
    SKL_TEST_REQUIRE(t, init_status, rs0 > 0);
  }
  if (n0 > 1) {
    SKL_TEST_REQUIRE(t, init_status, cs0 > 0);
  }
  if (m0 > 1 && n0 > 1) {
    if (rs0 >= cs0) {
      SKL_TEST_REQUIRE(t, init_status, rs0 >= (n0 - 1) * cs0 + 1);
    } else {
      SKL_TEST_REQUIRE(t, init_status, cs0 >= (m0 - 1) * rs0 + 1);
    }
  }

  size_t block_min_len = (m0 - 1) * rs0 + (n0 - 1) * cs0 + 1;
  if (m1 > 1 && n1 > 0) {
    SKL_TEST_REQUIRE(t, init_status, rs1 >= block_min_len);
  }
  if (m1 > 0 && n1 > 1) {
    SKL_TEST_REQUIRE(t, init_status, cs1 >= block_min_len);
  }
  if (m1 > 1 && n1 > 1) {
    if (rs1 >= cs1) {
      SKL_TEST_REQUIRE(t, init_status, rs1 >= (n1 - 1) * cs1 + block_min_len);
    } else {
      SKL_TEST_REQUIRE(t, init_status, cs1 >= (m1 - 1) * rs1 + block_min_len);
    }
  }
}

/**
 * @brief Check for clobbered elements in a packed matrix
 *
 * @param t - Test context.
 * @param c_type_size - Size of the elements of C_pack0 and C_pack1 in bytes.
 * @param c_pack_len - Number of elements in the arrays for C_pack0 and C_pack1.
 * @param m0 - Number of rows in each block of C_pack0 and C_pack1.
 * @param n0 - Number of columns in each block of C_pack0 and C_pack1.
 * @param m1 - Number of block-rows in C_pack0 and C_pack1.
 * @param n1 - Number of block-columns in C_pack0 and C_pack1.
 * @param c_pack0 - Pointer to matrix C_pack0.
 * @param c_pack1 - Pointer to matrix C_pack1.
 * @param rsc0 - Row stride within each block of C_pack0 and C_pack1 in
 *               elements.
 * @param csc0 - Column stride within each block of C_pack0 and C_pack1 in
 *               elements.
 * @param rsc1 - Row stride between blocks of C_pack0 and C_pack1 in elements.
 * @param csc1 - Column stride between blocks of C_pack0 and C_pack1 in
 *               elements.
 *
 * C_pack0 and C_pack1 are packed matrices stored in arrays with c_pack_len
 * elements of size c_type_size bytes. The elements of the matrices are indexed
 * by i1 * rsc1 + j1 * csc1 + i0 * rsc0 + j0 * csc0, 0 <= i1 < m1, 0 <= j1 < n1,
 * 0 <= i0 < m0, 0 <= j0 < n0. This function checks that the arrays pointed to
 * by c_pack0 and c_pack1 are identical outside of the above indices and updates
 * the verify_status of t.
 */
static inline void skl_test_check_matrix_clobbered_rcprc(
    skl_test_t *t, size_t c_type_size, size_t c_pack_len, size_t m0, size_t n0,
    size_t m1, size_t n1, void *c_pack0, void *c_pack1, size_t rsc0,
    size_t csc0, size_t rsc1, size_t csc1) {
  uint8_t *c_pack0_cast = (uint8_t *)c_pack0;
  uint8_t *c_pack1_cast = (uint8_t *)c_pack1;
  // When m0 == 1, rsc0 can be arbitrary. We adjust its value in this case to
  // reflect the length of a row. csc0, rsc1, and csc1 are adjusted similarly.
  size_t rsc0_adj = m0 == 1 ? (n0 - 1) * csc0 + 1 : rsc0;
  size_t csc0_adj = n0 == 1 ? (m0 - 1) * rsc0 + 1 : csc0;
  size_t c_block_min_len = (m0 - 1) * rsc0_adj + (n0 - 1) * csc0_adj + 1;
  size_t rsc1_adj = m1 == 1 ? (n1 - 1) * csc1 + c_block_min_len : rsc1;
  size_t csc1_adj = n1 == 1 ? (m1 - 1) * rsc1 + c_block_min_len : csc1;
  // Each block of C_pack{0,1} lies in an ambient m0_amb x n0_amb row- or
  // column-major matrix with row and column strides rsc0_amb and csc0_amb.
  // Let bsc1 be the block stride of C_pack{0,1}, i.e. the distance from one
  // block to the next in memory, in elements; bsc1 varies from block to block.
  // Lastly, set layoutc0_row_major to true if blocks of C_pack{0,1} are row
  // major and false otherwise. Similarly, set layoutc1_row_major to true if
  // C_pack{0,1} is block-row-major and false otherwise.
  size_t m0_amb = 0;
  size_t n0_amb = 0;
  size_t rsc0_amb = 0;
  size_t csc0_amb = 0;
  size_t bsc1 = 0;
  bool layoutc0_row_major = rsc0_adj >= csc0_adj;
  bool layoutc1_row_major = rsc1_adj >= csc1_adj;
  if (layoutc0_row_major) {
    rsc0_amb = rsc0_adj;
    csc0_amb = 1;
    m0_amb = m0;
    n0_amb = rsc0_amb;
  } else {
    rsc0_amb = 1;
    csc0_amb = csc0_adj;
    m0_amb = csc0_amb;
    n0_amb = n0;
  }

  // Check each block.
  for (size_t i1 = 0; i1 < m1; ++i1) {
    for (size_t j1 = 0; j1 < n1; ++j1) {
      // Check the first c_block_min_len elements.
      for (size_t i0 = 0; i0 < m0_amb; ++i0) {
        for (size_t j0 = 0; j0 < n0_amb; ++j0) {
          if (layoutc0_row_major && j0 % csc0_adj == 0 && j0 < n0 * csc0_adj) {
            continue;
          }
          if (!layoutc0_row_major && i0 % rsc0_adj == 0 && i0 < m0 * rsc0_adj) {
            continue;
          }
          if (i0 * rsc0_amb + j0 * csc0_amb >= c_block_min_len) {
            continue;
          }

          size_t idx =
              i1 * rsc1_adj + j1 * csc1_adj + i0 * rsc0_amb + j0 * csc0_amb;
          int64_t element0 = 0;
          int64_t element1 = 0;
          memcpy(&element0, &(c_pack0_cast[c_type_size * idx]), c_type_size);
          memcpy(&element1, &(c_pack1_cast[c_type_size * idx]), c_type_size);
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
      if (layoutc1_row_major) {
        bsc1 = j1 < n1 - 1 ? csc1_adj : rsc1_adj - (n1 - 1) * csc1_adj;
      } else {
        bsc1 = i1 < m1 - 1 ? rsc1_adj : csc1_adj - (m1 - 1) * rsc1_adj;
      }
      if (i1 == m1 - 1 && j1 == n1 - 1) {
        bsc1 = c_block_min_len;
      }
      for (size_t i = c_block_min_len; i < bsc1; ++i) {
        size_t idx = i1 * rsc1_adj + j1 * csc1_adj + i;
        int64_t element0 = 0;
        int64_t element1 = 0;
        memcpy(&element0, &(c_pack0_cast[c_type_size * idx]), c_type_size);
        memcpy(&element1, &(c_pack1_cast[c_type_size * idx]), c_type_size);
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
  size_t c_pack_min_len =
      m1 == 0 || n1 == 0
          ? 0
          : (m1 - 1) * rsc1_adj + (n1 - 1) * csc1_adj + c_block_min_len;
  for (size_t idx = c_pack_min_len; idx < c_pack_len; ++idx) {
    int64_t element0 = 0;
    int64_t element1 = 0;
    memcpy(&element0, &(c_pack0_cast[c_type_size * idx]), c_type_size);
    memcpy(&element1, &(c_pack1_cast[c_type_size * idx]), c_type_size);
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
