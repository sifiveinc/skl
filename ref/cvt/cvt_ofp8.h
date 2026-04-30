// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Convert a single OFP8 E4M3 value to IEEE FP32.
 *
 * @param in - Input OFP8 E4M3 value, represented as an 8-bit unsigned integer.
 * @return The converted IEEE FP32 value.
 */
float skl_cvt_f8e4m3_f32(uint8_t in);

/** @brief Convert a single OFP8 E5M2 value to IEEE FP32.
 *
 * @param in - Input OFP8 E5M2 value, represented as an 8-bit unsigned integer.
 * @return The converted IEEE FP32 value.
 */
float skl_cvt_f8e5m2_f32(uint8_t in);

/** @brief Convert a single IEEE FP32 value to OFP8 E4M3.
 *
 * @param in - Input IEEE FP32 value.
 * @param is_sat - If true, infinite results are clamped to the maximum finite
 * value of the same sign; if false, NaN is returned for overflow.
 * @return The converted OFP8 E4M3 value, represented as an 8-bit unsigned
 * integer.
 */
uint8_t skl_cvt_f32_f8e4m3(float in, bool is_sat);

/** @brief Convert a single IEEE FP32 value to OFP8 E5M2.
 *
 * @param in - Input IEEE FP32 value.
 * @param is_sat - If true, infinite results are clamped to the maximum finite
 * value of the same sign; if false, infinity is preserved.
 * @return The converted OFP8 E5M2 value, represented as an 8-bit unsigned
 * integer.
 */
uint8_t skl_cvt_f32_f8e5m2(float in, bool is_sat);

/** @brief Convert IEEE FP32 array to OFP8 E4M3 array (reference
 * implementation).
 *
 * @param pDst - Output array for converted OFP8 E4M3 values.
 * @param pSrc - Input array of IEEE FP32 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_f32_f8e4m3_ref(uint8_t *pDst, const float *pSrc,
                            float scaling_factor, size_t n);

/** @brief Convert IEEE FP32 array to OFP8 E4M3 array with saturation
 * (reference implementation).
 *
 * @param pDst - Output array for converted OFP8 E4M3 values.
 * @param pSrc - Input array of IEEE FP32 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_sat_f32_f8e4m3_ref(uint8_t *pDst, const float *pSrc,
                                float scaling_factor, size_t n);

/** @brief Convert IEEE FP32 array to OFP8 E5M2 array (reference
 * implementation).
 *
 * @param pDst - Output array for converted OFP8 E5M2 values.
 * @param pSrc - Input array of IEEE FP32 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_f32_f8e5m2_ref(uint8_t *pDst, const float *pSrc,
                            float scaling_factor, size_t n);

/** @brief Convert IEEE FP32 array to OFP8 E5M2 array with saturation
 * (reference implementation).
 *
 * @param pDst - Output array for converted OFP8 E5M2 values.
 * @param pSrc - Input array of IEEE FP32 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_sat_f32_f8e5m2_ref(uint8_t *pDst, const float *pSrc,
                                float scaling_factor, size_t n);

/** @brief Convert BF16 array to OFP8 E4M3 array (reference implementation).
 *
 * @param pDst - Output array for converted OFP8 E4M3 values.
 * @param pSrc - Input array of BF16 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_bf16_f8e4m3_ref(uint8_t *pDst, const __bf16 *pSrc,
                             float scaling_factor, size_t n);

/** @brief Convert BF16 array to OFP8 E4M3 array with saturation (reference
 * implementation).
 *
 * @param pDst - Output array for converted OFP8 E4M3 values.
 * @param pSrc - Input array of BF16 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_sat_bf16_f8e4m3_ref(uint8_t *pDst, const __bf16 *pSrc,
                                 float scaling_factor, size_t n);

/** @brief Convert BF16 array to OFP8 E5M2 array (reference implementation).
 *
 * @param pDst - Output array for converted OFP8 E5M2 values.
 * @param pSrc - Input array of BF16 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_bf16_f8e5m2_ref(uint8_t *pDst, const __bf16 *pSrc,
                             float scaling_factor, size_t n);

/** @brief Convert BF16 array to OFP8 E5M2 array with saturation (reference
 * implementation).
 *
 * @param pDst - Output array for converted OFP8 E5M2 values.
 * @param pSrc - Input array of BF16 values.
 * @param scaling_factor - Scaling factor applied to each input element before
 * conversion.
 * @param n - Number of elements to convert.
 */
void skl_cvt_sat_bf16_f8e5m2_ref(uint8_t *pDst, const __bf16 *pSrc,
                                 float scaling_factor, size_t n);

/** @brief Convert OFP8 E4M3 array to BF16 array (reference implementation).
 *
 * @param pDst - Output array for converted BF16 values.
 * @param pSrc - Input array of OFP8 E4M3 values.
 * @param n - Number of elements to convert.
 */
void skl_cvt_f8e4m3_bf16_ref(__bf16 *pDst, const uint8_t *pSrc, size_t n);

/** @brief Convert OFP8 E5M2 array to BF16 array (reference implementation).
 *
 * @param pDst - Output array for converted BF16 values.
 * @param pSrc - Input array of OFP8 E5M2 values.
 * @param n - Number of elements to convert.
 */
void skl_cvt_f8e5m2_bf16_ref(__bf16 *pDst, const uint8_t *pSrc, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
