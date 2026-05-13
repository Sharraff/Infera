#include "../kernels.h"
#include "../internal/half_bf16_convert.h"

void add_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    uint16_t *o = (uint16_t *)out;
    const uint16_t *x = (const uint16_t *)a;
    const uint16_t *y = (const uint16_t *)b;

    for (int64_t i = 0; i < n; i++) {
        float fx = infera_f16_bits_to_f32(x[i]);
        float fy = infera_f16_bits_to_f32(y[i]);
        o[i] = infera_f32_to_f16_bits(fx + fy);
    }
}

void sub_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    uint16_t *o = (uint16_t *)out;
    const uint16_t *x = (const uint16_t *)a;
    const uint16_t *y = (const uint16_t *)b;

    for (int64_t i = 0; i < n; i++) {
        float fx = infera_f16_bits_to_f32(x[i]);
        float fy = infera_f16_bits_to_f32(y[i]);
        o[i] = infera_f32_to_f16_bits(fx - fy);
    }
}

void mul_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    uint16_t *o = (uint16_t *)out;
    const uint16_t *x = (const uint16_t *)a;
    const uint16_t *y = (const uint16_t *)b;

    for (int64_t i = 0; i < n; i++) {
        float fx = infera_f16_bits_to_f32(x[i]);
        float fy = infera_f16_bits_to_f32(y[i]);
        o[i] = infera_f32_to_f16_bits(fx * fy);
    }
}

void div_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    uint16_t *o = (uint16_t *)out;
    const uint16_t *x = (const uint16_t *)a;
    const uint16_t *y = (const uint16_t *)b;

    for (int64_t i = 0; i < n; i++) {
        float fx = infera_f16_bits_to_f32(x[i]);
        float fy = infera_f16_bits_to_f32(y[i]);
        o[i] = infera_f32_to_f16_bits(fx / fy);
    }
}
