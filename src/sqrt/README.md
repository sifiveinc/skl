# Square-root Kernels

The ubiquitous square-root function.

It's implementation here serves several purposes:

1) to document some best-practice use of RVV's `vfrsqrt7.v` instruction, and

2) to demonstrate the variety of approaches available, and

3) to provide performance options for applications.

The performance of these kernels, in particular the `1u` and `2u` variants, may be better than using RVV's `vfsqrt.v` instruction directly, though vector register use will be higher.
Performance should be compared carefully on your target system.

## Kernel List

### F64 Kernels

```c
void skl_sqrt_f64_zve64d(double *out, const double *in, size_t n);
void skl_sqrt_1u_f64_zve64d(double *out, const double *in, size_t n);
```

### F32 Kernels

```c
void skl_sqrt_f32_zve32f(float *out, const float *in, size_t n);
void skl_sqrt_1u_f32_zve32f(float *out, const float *in, size_t n);
```

### F16 Kernels

```c
void skl_sqrt_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_sqrt_1u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);

```
### BF16 Kernels

```c
void skl_sqrt_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);
void skl_sqrt_2u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);
void skl_sqrt_bf16_zvfbfmin(__bf16 *out, const __bf16 *in, size_t n);
void skl_sqrt_2u_bf16_zvfbfmin(__bf16 *out, const __bf16 *in, size_t n);
void skl_sqrt_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);
void skl_sqrt_1u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);
```
