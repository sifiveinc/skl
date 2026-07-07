# Logistic (Sigmoid) Kernels

The Logistic function is a widely-used sigmoid activation function in neural networks that maps inputs to values between 0 and 1.
It is primarily used in multi-label classification and logistic regression models.

Sometimes known simply as "sigmoid", the Logistic function computes the following elementwise function:


```
logistic(x) = 1 / (1 + e^(-x))
```

## Kernel Naming Convention

The kernel names follow the pattern: `skl_logistic_<accuracy>_<type>_<isa>`

where:

- `<accuracy>`: denotes the maximum absolute error in ULP (e.g., `3u` for 3 ULP).
- `<type>`: denotes the type of the input and output arrays.
- `<isa>`: denotes the instruction set architecture extension(s) the kernel requires.

## Constraints

- Input and output arrays may overlap only for in-place computation.
- For maximum performance, input and output arrays should be naturally aligned.

## Kernel List

- FP32 APIs

```c
void skl_logistic_3u_f32_zve32f(float *out, const float *in, size_t n);
void skl_logistic_4u_f32_xsfvfexpa(float *out, const float *in, size_t n);
void skl_logistic_5u_f32_xsfvfexp32e(float *out, const float *in, size_t n);
```

- FP16 APIs

```c
void skl_logistic_3u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_logistic_5u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n);
```

- BF16 APIs

```c
void skl_logistic_2u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);
void skl_logistic_3u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);
void skl_logistic_2u_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *in, size_t n);
void skl_logistic_5u_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);
```
