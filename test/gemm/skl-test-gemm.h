#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void gemm_rcprc_check_clobbered(skl_test_t *t, size_t c_type_size,
                                size_t c_pack_len, size_t m0, size_t n0,
                                size_t m1, size_t n1, void *c_pack, size_t rsc0,
                                size_t csc0, size_t rsc1, size_t csc1,
                                void *ref_c) {
  uint8_t *c_pack_cast = (uint8_t *)c_pack;
  uint8_t *ref_c_cast = (uint8_t *)ref_c;
  // When m0 == 1, rsc0 can be arbitrary. We adjust its value in this case to
  // reflect the length of a row. csc0, rsc1, and csc1 are adjusted similarly.
  size_t rsc0_adj = m0 == 1 ? (n0 - 1) * csc0 + 1 : rsc0;
  size_t csc0_adj = n0 == 1 ? (m0 - 1) * rsc0 + 1 : csc0;
  size_t c_block_min_len = (m0 - 1) * rsc0_adj + (n0 - 1) * csc0_adj + 1;
  size_t rsc1_adj = m1 == 1 ? (n1 - 1) * csc1 + c_block_min_len : rsc1;
  size_t csc1_adj = n1 == 1 ? (m1 - 1) * rsc1 + c_block_min_len : csc1;
  // Each block of C_pack lies in an ambient m0_amb x n0_amb row- or
  // column-major matrix with row and column strides rsc0_amb and csc0_amb.
  // Let bsc1 be the block stride of C_pack, i.e. the distance from one block to
  // the next in memory, in elements; bsc1 varies from block to block.
  // Lastly, set layoutc0_row_major to true if blocks of C_pack are row major
  // and false otherwise. Similarly, set layoutc1_row_major to true if C_pack is
  // block-row-major and false otherwise.
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
          int64_t initial = 0;
          int64_t result = 0;
          memcpy(&initial, &(ref_c_cast[c_type_size * idx]), c_type_size);
          memcpy(&result, &(c_pack_cast[c_type_size * idx]), c_type_size);
          if (result != initial) {
            SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                         "result [%zu, %zu, %zu, %zu] clobbered\n", i1, j1, i0,
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
        // NaNs always compare as false, so do a bitwise comparison.
        int64_t initial = 0;
        int64_t result = 0;
        memcpy(&initial, &(ref_c_cast[c_type_size * idx]), c_type_size);
        memcpy(&result, &(c_pack_cast[c_type_size * idx]), c_type_size);
        if (result != initial) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "result [%zu, %zu, %zu] clobbered\n", i1, j1, i);
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
    // NaNs always compare as false, so do a bitwise comparison.
    int64_t initial = 0;
    int64_t result = 0;
    memcpy(&initial, &(ref_c_cast[c_type_size * idx]), c_type_size);
    memcpy(&result, &(c_pack_cast[c_type_size * idx]), c_type_size);
    if (result != initial) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "result [%zu] clobbered\n", idx);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

