# Softmax Kernels

The Softmax function computes the following normalization:

```
softmax(x) = exp(xᵢ) / sum(exp(xⱼ))
```

Output values are between zero and one, inclusive, and sum to one[^1].

The implementations here use a "stable" algorithm to avoid spurious over- and underflow,
and support a `beta` parameter to control the base of the exponentiation:

```
softmax(x,β) = exp(β(xᵢ - M)) / sum(exp(β(xⱼ- M)))
  where
    M = max(x),
    exp(β*a) ~ pow(γ,a), β = log(γ)
```

Numerical reproducibility is not guaranteed between ISA kernels or different array/matrix sizes.

[^1]: Or, with the occurence of floating-point rounding, close to one.

## Kernel Naming Convention

The kernel names follow the pattern: `skl_softmax_<datatype>_<isa>`

where:

- `<datatype>`: further decomposed as `<type>[<axis-major>]`,
  where `<type>` is one of the [type specifiers](../README.md#type-specifier-convention) defined in the SKL API design document and denotes the floating-point type of the inputs and outputs,
  and `<axis-major>` is an optional specification denoting a 2D Softmax.
  It can be `r` for row-major storage, `c` for column-major storage, or absent for a 1D Softmax.
- `<isa>`: denotes the instruction set architecture extension(s) the kernel requires.


## Constraints

- Input and output arrays may overlap only for in-place computation, that is, when the output array is exactly the input array.
- For maximum performance, input and output arrays should be naturally aligned.
- The `beta` parameter must be greater than or equal to zero.


## Kernel List

### One-dimensional Softmax

- F32 APIs

```c
void skl_softmax_f32_zve32f(float *y, const float *x, float beta, size_t n);
void skl_softmax_f32_xsfvfexpa(float *y, const float *x, float beta, size_t n);
void skl_softmax_f32_xsfvfexp32e(float *y, const float *x, float beta, size_t n);
```

- F16 APIs

```c
void skl_softmax_f16_zvfh(_Float16 *y, const _Float16 *x,
                          _Float16 beta, size_t n);
void skl_softmax_f16_xsfvfexpa_zvfh(_Float16 *y, const _Float16 *x,
                                    _Float16 beta, size_t n);
void skl_softmax_f16_xsfvfexp16e(_Float16 *y, const _Float16 *x,
                                 _Float16 beta, size_t n);
```

- BF16 APIs

```c
void skl_softmax_bf16_zve32f(__bf16 *y, const __bf16 *x, __bf16 beta, size_t n);
void skl_softmax_bf16_zvfbfmin(__bf16 *y, const __bf16 *x, __bf16 beta, size_t n);
void skl_softmax_bf16_xsfvfbfa(__bf16 *y, const __bf16 *x, __bf16 beta, size_t n);
void skl_softmax_bf16_xsfvfexpa_zvfbfmin(__bf16 *y, const __bf16 *x,
                                         __bf16 beta, size_t n);`
void skl_softmax_bf16_xsfvfexpa_xsfvfbfa(__bf16 *y, const __bf16 *x,
                                         __bf16 beta, size_t n);
void skl_softmax_bf16_xsfvfbfexp16e_zvfbfmin(__bf16 *y, const __bf16 *x,
                                             __bf16 beta, size_t n);`
void skl_softmax_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *y, const __bf16 *x,
                                             __bf16 beta, size_t n);
void skl_softmax_bf16_xsfvfexp32e_zvfbfmin(__bf16 *y, const __bf16 *x,
                                           __bf16 beta, size_t n);`
``

### Two-dimensional Softmax

The two-dimensional kernels add function parameters for row stride of the row-major input and output matrices and for the number of rows.

- F32 APIs

```c
void skl_softmax_f32r_zve32f(float *s, size_t rss, const float *a, size_t rsa,
                             float beta, size_t m, size_t n);
void skl_softmax_f32r_xsfvfexpa(float *s, size_t rss, const float *a, size_t rsa,
                                float beta, size_t m, size_t n);
void skl_softmax_f32r_xsfvfexp32e(float *s, size_t rss, const float *a, size_t rsa,
                                  float beta, size_t m, size_t n);
```

Two-dimensional kernels are not yet defined for F16 or BF16 datatypes or column-major storage.
