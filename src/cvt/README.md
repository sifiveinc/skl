# OFP (*Open Compute Project* Floating Point) Conversion Kernels

The OFP conversion kernels provide vectorized conversion functions between various floating-point formats and 8-bit/4-bit OFP formats using RISC-V Zvfofp8min and Zvfofp4min extensions. These kernels enable efficient data type conversions for neural network workloads that utilize low-precision floating-point formats.

For more details on OFP formats, see the [OCP Microscaling Formats (MX) v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

## OFP8 Conversion Kernels (Zvfofp8min)

The OFP8 conversion functions support conversions between standard floating-point formats (F32, BF16) and 8-bit OFP formats (E4M3, E5M2). Note that not all combinations support bidirectional conversion.

### Narrowing Conversions (Standard → OFP8)

Narrowing conversions reduce precision and dynamic range from standard formats to 8-bit OFP formats. A scaling factor is required.

**Basic conversion (F32 to E4M3):**
```c
/* in cvt_zvfofp8min.h */

/** F32 to E4M3 conversion with scaling.
 *
 * Converts F32 values to 8-bit E4M3 OFP format with scaling factor (pass 1.0f for no scaling).
 * Infinite results are preserved as infinity in the target format.
 */
void skl_cvt_f32_f8e4m3_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                 float scaling_factor, size_t n);
```

**Saturating conversion (F32 to E4M3):**
```c
/** F32 to E4M3 conversion with scaling and saturation.
 *
 * Converts F32 values to 8-bit E4M3 OFP format with scaling factor (pass 1.0f for no scaling).
 * Infinite results are clamped to the maximum finite value of the same sign.
 */
void skl_cvt_sat_f32_f8e4m3_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                     float scaling_factor, size_t n);
```

### Widening Conversions (OFP8 → Standard)

Widening conversions increase precision and dynamic range from 8-bit OFP formats to standard formats. No scaling factor is required.

**E4M3 to BF16 conversion:**
```c
/** E4M3 to BF16 conversion.
 *
 * Converts 8-bit E4M3 OFP values to BF16 format with full precision preservation.
 * No scaling factor is needed as this is a widening conversion.
 */
void skl_cvt_f8e4m3_bf16_zvfofp8min(__bf16 *pDst, const uint8_t *pSrc, size_t n);
```

### Supported OFP8 Conversion Types

| Source | Destination | Function | Saturating Variant | Scaling Required |
|--------|-------------|----------|-------------------|------------------|
| F32 | E4M3 | `skl_cvt_f32_f8e4m3_zvfofp8min` | `skl_cvt_sat_f32_f8e4m3_zvfofp8min` | Yes |
| F32 | E5M2 | `skl_cvt_f32_f8e5m2_zvfofp8min` | `skl_cvt_sat_f32_f8e5m2_zvfofp8min` | Yes |
| BF16 | E4M3 | `skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin` | `skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin` | Yes |
| BF16 | E5M2 | `skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin` | `skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin` | Yes |
| E4M3 | BF16 | `skl_cvt_f8e4m3_bf16_zvfofp8min` | N/A | No |
| E5M2 | BF16 | `skl_cvt_f8e5m2_bf16_zvfofp8min` | N/A | No |

## OFP4 Conversion Kernels (Zvfofp4min)

The OFP4 conversion functions support widening conversions from 4-bit E2M1 format to 8-bit E4M3 format.

### Widening Conversions (OFP4 → OFP8)

**E2M1 to E4M3 conversion:**
```c
/* in cvt_zvfofp4min.h */

/** E2M1 to E4M3 conversion.
 *
 * Converts packed 4-bit E2M1 OFP values to 8-bit E4M3 OFP format.
 * Input format: Each byte contains two 4-bit values (lower 4 bits first, upper 4 bits second).
 * Output length n can be odd; the function handles partial pairs correctly.
 */
void skl_cvt_f4e2m1_f8e4m3_zvfofp4min(uint8_t *pDst, const uint8_t *pSrc, size_t n);
```

### Memory Layout Requirements

**Input Array:**
- Must be byte-aligned
- Each byte contains two packed 4-bit E2M1 values:
  ```
  Byte layout: [upper 4 bits: E2M1[1]] [lower 4 bits: E2M1[0]]
  ```

**Output Array:**
- Length `n` can be odd; the function handles partial pairs at the end
- Each output byte contains one E4M3 value

## Kernel Naming Convention

The Kernel names follow the pattern: `skl_cvt_[sat_]<src>_<dst>_<arch>`

- **`sat_`**: Optional prefix indicating saturating conversion (clamps infinities)
- **`<src>`**: Source format (f32, bf16, f8e4m3, f8e5m2, f4e2m1)
- **`<dst>`**: Destination format (f8e4m3, f8e5m2, bf16)
- **`<arch>`**: Target architecture extension (zvfofp8min, zvfofp8min_zvfbfmin, zvfofp4min)

## Implementation Details

### Scaling Factor Optimization
- Special-case optimization when `scaling_factor == 1.0f`
- Avoids unnecessary floating-point multiplication when no scaling is needed
- Scaling applied before format conversion for maximum precision

### Extension Dependencies
- **Zvfofp8min**: Required for all OFP8 conversion functions
- **Zvfbfmin**: Required for narrowing conversions from BF16 to OFP8
  - **Note**: When `scaling_factor != 1.0f`, BF16 values must be converted to F32 format to perform the scaling multiplication before conversion to OFP8. This intermediate F32 conversion requires the Zvfbfmin extension for BF16 arithmetic operations.
- **Zvfofp4min**: Required for OFP4 conversion functions
- Functions are conditionally compiled based on extension availability
