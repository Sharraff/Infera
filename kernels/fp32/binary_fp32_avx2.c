/*
 * FP32 elementwise kernels using AVX2 (8 floats / step).
 * This translation unit should be compiled with -mavx2 -mfma on x86-64 hosts.
 */
#include "../kernels.h"

#include <immintrin.h>

void add_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    float *o = (float *)out;
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_add_ps(vx, vy));
    }
    for (; i < n; i++)
        o[i] = x[i] + y[i];
}

void sub_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    float *o = (float *)out;
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_sub_ps(vx, vy));
    }
    for (; i < n; i++)
        o[i] = x[i] - y[i];
}

void mul_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    float *o = (float *)out;
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_mul_ps(vx, vy));
    }
    for (; i < n; i++)
        o[i] = x[i] * y[i];
}

void div_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    float *o = (float *)out;
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_div_ps(vx, vy));
    }
    for (; i < n; i++)
        o[i] = x[i] / y[i];
}
