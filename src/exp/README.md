# Exponential Kernels

## Kernel Naming Convention

The kernel names for the pattern: `skl_exp_<desc>_<type>_<isa>`

where:

- `<desc>` is a compact description of the "valid" input domain,
  result accuracy, and behavior for invalid inputs.  See complete
  grammar below.
- `<type>` is one of the [type
  specifiers](../README.md#type-specifier-convention) defined in the
  SKL API design document
- `<isa>` denotes the instruction set architecture extension the
  kernel requires.

## Kernel Descriptor

There are many ways to compute the exponential function, in both
software and hardware.  Such differences affect the accuracy of
results, behavior for edge cases, and, together with application
constraints, ultimately the circumstances in which a particular
implementation should be used.

While individual kernels document relevant characteristics in their
respective headers, it is nevertheless useful to concisely convey some
of these characteristics directly in the kernel's name, which, amongst
other benefits, allows for the definition of kernels having different
characteristics but the same `<type>_<isa>`.

The descriptors used here follow roughly the following grammar:

```
DESC     := ULP                         // minimally contains accuracy
          | ULP DOMAIN                  // optionally domain where valid
ULP      := FIXED 'u'                   // accuracy encoded as fixed-point
DOMAIN   := COMPARE SAT                 // domain encodes valid inputs and
          | INTERVAL SAT                //   saturation behavior for invalid,
          | SAT INTERVAL SAT            //   behavior for lesser and greater
COMPARE  := CMP SIGNED                  // comparison against signed number
CMP      := "lt" | "le" | "ge" | "gt"   // standard comparators
INTERVAL := RANGE | ABSOLUTE            // literal range or relative to |x|
RANGE    := 'c' PAIR 'c' | 'o' PAIR 'o' // closed or open interval
PAIR     := SIGNED 'x' SIGNED           // encoded pair
ABSOLUTE := 'a' COMPARE 'a'             // |x| satisfies comparison
SAT      := SIGNED | 's' SIGNED         // 's' if needed to disambiguate
SIGNED   := NUMBER | 'n' NUMBER         // leading 'n' for sign
NUMBER   := FIXED | FLOAT | CONST
FIXED    := DIGIT+ | DIGIT+ 'p' DIGIT+  // 'p' used for decimal point
FLOAT    := HEX+ 'P' SINT               // hex float, implicit leading "0x1."
CONST    := "inf" | 'U'                 // "infinity" or "undefined"
SINT     := DIGIT+ | 'n' DIGIT+         // signed integer
DIGIT    := [0123456789]
HEX      := DIGIT | [abcdef]
```

### Examples

- "1u" -> 1 ULP accuracy over all values in domain.
- "3p16u" -> 3.16 ULP accuracy over all values in domain.
- "1u0alt64ainf" -> 1 ULP accuracy for inputs `|x| < 64`, lesser
  inputs return 0, and greater inputs return infinity.
- "1p132ugen37P3s0" -> 1.132 ULP accuracy for inputs `x ≥ -0x1.37p3`,
  lesser inputs saturate to zero.

## Constraints

- Input and output arrays may overlap only for in-place computation,
  that is, when the output array is exactly the input array.
- For maximum performance, input and output arrays should be naturally
  aligned.


## Kernel List

### F32 Kernels

```c
void skl_exp_1u_f32_zve32f(float *out, const float *in, size_t n);
void skl_exp_1u_f32_xsfvfexpa(float *out, const float *in, size_t n);
void skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa(float *out, const float *in, size_t n);
void skl_exp_2p398u0alt64ainf_f32_xsfvfexp32e(float *out, const float *in, size_t n);
void skl_exp_5p32u_f32_xsfvfexp32e(float *out, const float *in, size_t n);
```

### F16 Kernels

```c
void skl_exp_1u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_exp_1u_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n);
void skl_exp_3p16u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n);
```

### BF16 Kernels

```c
void skl_exp_1u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);
void skl_exp_1u_bf16_zvfbfmin(__bf16 *out, const __bf16 *in, size_t n);
void skl_exp_1u_bf16_xsfvfexpa_zvfbfmin(__bf16 *out, const __bf16 *in, size_t n);
void skl_exp_1u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);
void skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e(__bf16 *out, const __bf16 *in, size_t n);
```
