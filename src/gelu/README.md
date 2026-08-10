# GELU Kernels

The Gaussian Error Linear Unit (GELU) activation function, used in
many transformer models, computes the following:

```
  GELU = x Φ(x)
       = x P(X≤x)
       = 0.5 x (1 + erf(x/sqrt(2)))
```

## Constraints

- Input and output arrays may overlap only for in-place computation,
  that is, when the output array is exactly the input array.
- For maximum performance, input and output arrays should be naturally
  aligned.


## Kernel List

Four functions are defined, each with different accuracy
characteristics.  Generally, the faster kernels have less accuracy.

```c
void skl_gelu_p9_f32_zve32f(float *dst, const float *src, size_t n);
void skl_gelu_p13_f32_zve32f(float *dst, const float *src, size_t n);
void skl_gelu_p17_f32_zve32f(float *dst, const float *src, size_t n);
void skl_gelu_rat_f32_zve32f(float *dst, const float *src, size_t n);
```


## Notes

Functions return NaN for input equal to negative infinity.
