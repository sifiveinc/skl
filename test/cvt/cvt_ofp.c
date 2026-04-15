// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the cvt_ofp test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. cvt_ofp_zvfofp8min.c).
 */

#include "cvt_ofp.h"
#include "skl-test-driver.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl-ref.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void cvt_ofp4x2_f8e4m3(uint8_t in, uint8_t *out0, uint8_t *out1) {

  uint8_t in_0 = in & 0xFU;
  uint8_t in_1 = in >> 4U;
  *out0 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_0), false);
  *out1 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_1), false);
}

static void golden_cvt_f4e2m1_f8e4m3(uint8_t *pDst, const uint8_t *pSrc,
                                     size_t n) {
  for (size_t i = 0; i < n / 2 * 2; i += 2) {
    cvt_ofp4x2_f8e4m3(pSrc[i / 2], pDst + i, pDst + i + 1);
  }
  if (n % 2 == 1) {
    uint8_t tmp;
    cvt_ofp4x2_f8e4m3(pSrc[n / 2], pDst + n - 1, &tmp);
  }
}

static void init_special_values_f32(float *data, size_t len) {
  size_t count = len < 3 ? len : 3;
  if (count > 0)
    data[0] = nanf("");
  if (count > 1)
    data[1] = INFINITY;
  if (count > 2)
    data[2] = -INFINITY;
}

static void init_special_values_bf16(__bf16 *data, size_t len) {
  size_t count = len < 3 ? len : 3;
  if (count > 0)
    data[0] = (__bf16)nanf("");
  if (count > 1)
    data[1] = (__bf16)INFINITY;
  if (count > 2)
    data[2] = (__bf16)-INFINITY;
}

static int init_input_f32(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(float)
  in = {.region = h->in_region,
        .len = h->len,
        .max = 512.0f,
        .min = -512.0f,
        .mode = SKL_TEST_RANDOM};
  SKL_TEST_BUF_CREATE(t, float, &in);
  h->in = in.data;
  init_special_values_f32(in.data, h->len);
  return 0;
}

static int init_input_bf16(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(__bf16)
  in = {.region = h->in_region,
        .len = h->len,
        .max = (__bf16)512.0f,
        .min = (__bf16)-512.0f,
        .mode = SKL_TEST_RANDOM};
  SKL_TEST_BUF_CREATE(t, __bf16, &in);
  h->in = in.data;
  init_special_values_bf16(in.data, h->len);
  return 0;
}

static int init_input_ofp8(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(uint8_t)
  in = {.region = h->in_region,
        .len = h->len,
        .max = 0,
        .min = 255,
        .mode = SKL_TEST_SEQ};
  SKL_TEST_BUF_CREATE(t, uint8_t, &in);
  h->in = in.data;
  return 0;
}

static int init_input_f4e2m1(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(uint8_t)
  in = {.region = h->in_region,
        .len = (h->len + 1) / 2,
        .max = 0,
        .min = 255,
        .mode = SKL_TEST_SEQ};
  SKL_TEST_BUF_CREATE(t, uint8_t, &in);
  h->in = in.data;
  return 0;
}

static int init_input_buffer(skl_test_t *t, cvt_ofp_t *h) {
  switch (h->in_type) {
  case F32:
    return init_input_f32(t, h);
  case BF16:
    return init_input_bf16(t, h);
  case F8E4M3:
  case F8E5M2:
    return init_input_ofp8(t, h);
  case F4E2M1:
    return init_input_f4e2m1(t, h);
  default:
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "Unsupported input type\n");
    return 1;
  }
}

static int init_output_bf16(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(__bf16)
  out = {.region = h->out_region,
         .len = h->len,
         .data = NULL,
         .mode = SKL_TEST_SEQ,
         .min = (__bf16)0.0f,
         .max = (__bf16)(h->len - 1)};
  if (h->len > 0) {
    SKL_TEST_BUF_CREATE(t, __bf16, &out);
    h->out = out.data;
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(__bf16)) : NULL;
  }
  return 0;
}

static int init_output_ofp8(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(uint8_t)
  out = {.region = h->out_region,
         .len = h->len,
         .data = NULL,
         .mode = SKL_TEST_SEQ,
         .min = 0,
         .max = (uint8_t)(h->len - 1)};
  if (h->len > 0) {
    SKL_TEST_BUF_CREATE(t, uint8_t, &out);
    h->out = out.data;
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(uint8_t)) : NULL;
  }
  return 0;
}

static int init_output_buffer(skl_test_t *t, cvt_ofp_t *h) {
  switch (h->out_type) {
  case BF16:
    return init_output_bf16(t, h);
  case F8E4M3:
  case F8E5M2:
    return init_output_ofp8(t, h);
  default:
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "Unsupported output type\n");
    return 1;
  }
}

void cvt_ofp_init(skl_test_t *t) {
  cvt_ofp_t *h = (cvt_ofp_t *)t->harness;
  if (init_input_buffer(t, h) != 0) {
    t->status.init_status = SKL_TEST_FAIL;
    return;
  }
  if (init_output_buffer(t, h) != 0) {
    t->status.init_status = SKL_TEST_FAIL;
    return;
  }
  t->status.init_status = SKL_TEST_PASS;
}

static void compute_reference_f32_to_ofp8(cvt_ofp_t *h) {
  float *in = (float *)h->in;
  uint8_t *ref = (uint8_t *)h->ref;
  for (size_t i = 0; i < h->len; ++i) {
    ref[i] = (h->out_type == F8E4M3)
                 ? skl_cvt_f32_f8e4m3(in[i] * h->scale, h->saturation)
                 : skl_cvt_f32_f8e5m2(in[i] * h->scale, h->saturation);
  }
}

static void compute_reference_bf16_to_ofp8(cvt_ofp_t *h) {
  __bf16 *in = (__bf16 *)h->in;
  uint8_t *ref = (uint8_t *)h->ref;
  for (size_t i = 0; i < h->len; ++i) {
    ref[i] = (h->out_type == F8E4M3)
                 ? skl_cvt_f32_f8e4m3((float)in[i] * h->scale, h->saturation)
                 : skl_cvt_f32_f8e5m2((float)in[i] * h->scale, h->saturation);
  }
}

static void compute_reference_ofp8_to_bf16(cvt_ofp_t *h) {
  uint8_t *in = (uint8_t *)h->in;
  __bf16 *ref = (__bf16 *)h->ref;
  for (size_t i = 0; i < h->len; ++i) {
    ref[i] = (h->in_type == F8E4M3) ? (__bf16)skl_cvt_f8e4m3_f32(in[i])
                                    : (__bf16)skl_cvt_f8e5m2_f32(in[i]);
  }
}

static void compute_reference_f4e2m1_to_f8e4m3(cvt_ofp_t *h) {
  golden_cvt_f4e2m1_f8e4m3((uint8_t *)h->ref, (uint8_t *)h->in, h->len);
}

static void compute_reference(cvt_ofp_t *h) {
  if (h->in_type == F32 && (h->out_type == F8E4M3 || h->out_type == F8E5M2)) {
    compute_reference_f32_to_ofp8(h);
    return;
  }
  if (h->in_type == BF16 && (h->out_type == F8E4M3 || h->out_type == F8E5M2)) {
    compute_reference_bf16_to_ofp8(h);
    return;
  }
  if ((h->in_type == F8E4M3 || h->in_type == F8E5M2) && h->out_type == BF16) {
    compute_reference_ofp8_to_bf16(h);
    return;
  }
  if (h->in_type == F4E2M1 && h->out_type == F8E4M3) {
    compute_reference_f4e2m1_to_f8e4m3(h);
    return;
  }
}

static int compare_outputs_uint8(skl_test_t *t, cvt_ofp_t *h, size_t i) {
  uint8_t *out = (uint8_t *)h->out;
  uint8_t *ref = (uint8_t *)h->ref;
  if (out[i] != ref[i]) {
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "result [%zu] (%d) != reference (%d)\n",
                 i, out[i], ref[i]);
    return 1;
  }
  return 0;
}

static int compare_outputs_bf16(skl_test_t *t, cvt_ofp_t *h, size_t i) {
  __bf16 *out = (__bf16 *)h->out;
  __bf16 *ref = (__bf16 *)h->ref;

  uint16_t out_bits;
  uint16_t ref_bits;
  memcpy(&out_bits, &out[i], sizeof(uint16_t));
  memcpy(&ref_bits, &ref[i], sizeof(uint16_t));

  if (out_bits != ref_bits) {
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "result [%zu] (%A) != reference (%A)\n",
                 i, (float)out[i], (float)ref[i]);
    return 1;
  }
  return 0;
}

static int verify_outputs(skl_test_t *t, cvt_ofp_t *h) {
  for (size_t i = 0; i < h->len; ++i) {
    int result = 0;
    if (h->out_type == F8E4M3 || h->out_type == F8E5M2) {
      result = compare_outputs_uint8(t, h, i);
    } else if (h->out_type == BF16) {
      result = compare_outputs_bf16(t, h, i);
    }
    if (result != 0)
      return 1;
  }
  return 0;
}

void cvt_ofp_verify(skl_test_t *t) {
  cvt_ofp_t *h = (cvt_ofp_t *)t->harness;
  compute_reference(h);
  if (verify_outputs(t, h) != 0) {
    t->status.verify_status = SKL_TEST_FAIL;
  } else {
    t->status.verify_status = SKL_TEST_PASS;
  }
}

static const char *type_to_str(cvt_ofp_type_t type) {
  switch (type) {
  case F8E4M3:
    return "F8E4M3";
  case F8E5M2:
    return "F8E5M2";
  case F4E2M1:
    return "F4E2M1";
  case BF16:
    return "BF16";
  case F32:
    return "F32";
  default:
    return "UNKNOWN";
  }
}

void cvt_ofp_report(skl_test_t *t) {
  cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  size_t elements = h->len;
  float elements_per_cycle = (float)elements / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Input type: %s\n", type_to_str(h->in_type));
  INFO("Output type: %s\n", type_to_str(h->out_type));
  if (h->in_type > h->out_type) { // narrowing conversion
    INFO("Saturation: %s\n", h->saturation ? "yes" : "no");
    INFO("Scaling factor: %s\n", h->scale != 1.0f ? "!= 1.0f" : "1.0f");
  }
  INFO("Length: %zd\n", h->len);
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("Elements/Cycle: %f\n", elements_per_cycle);

  t->status.report_status = SKL_TEST_PASS;
}

static int cleanup_input_f32(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(float) in = {.region = h->in_region, .data = h->in};
  SKL_TEST_BUF_FREE(t, &in);
  return 0;
}

static int cleanup_input_bf16(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(__bf16) in = {.region = h->in_region, .data = h->in};
  SKL_TEST_BUF_FREE(t, &in);
  return 0;
}

static int cleanup_input_uint8(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(uint8_t) in = {.region = h->in_region, .data = h->in};
  SKL_TEST_BUF_FREE(t, &in);
  return 0;
}

static int cleanup_input_buffer(skl_test_t *t, cvt_ofp_t *h) {
  switch (h->in_type) {
  case F32:
    return cleanup_input_f32(t, h);
  case BF16:
    return cleanup_input_bf16(t, h);
  case F8E4M3:
  case F8E5M2:
  case F4E2M1:
    return cleanup_input_uint8(t, h);
  default:
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "Unsupported input type\n");
    return 1;
  }
}

static int cleanup_output_bf16(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(__bf16) out = {.region = h->out_region, .data = h->out};
  SKL_TEST_BUF_FREE(t, &out);
  return 0;
}

static int cleanup_output_uint8(skl_test_t *t, cvt_ofp_t *h) {
  SKL_TEST_BUFFER(uint8_t) out = {.region = h->out_region, .data = h->out};
  SKL_TEST_BUF_FREE(t, &out);
  return 0;
}

static int cleanup_output_buffer(skl_test_t *t, cvt_ofp_t *h) {
  switch (h->out_type) {
  case BF16:
    return cleanup_output_bf16(t, h);
  case F8E4M3:
  case F8E5M2:
    return cleanup_output_uint8(t, h);
  default:
    SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "Unsupported output type\n");
    return 1;
  }
}

void cvt_ofp_cleanup(skl_test_t *t) {
  cvt_ofp_t *h = (cvt_ofp_t *)t->harness;
  if (cleanup_input_buffer(t, h) != 0) {
    t->status.cleanup_status = SKL_TEST_FAIL;
    return;
  }
  if (cleanup_output_buffer(t, h) != 0) {
    t->status.cleanup_status = SKL_TEST_FAIL;
    return;
  }
  if (h->steps.verify)
    free(h->ref);
  t->status.cleanup_status = SKL_TEST_PASS;
}
