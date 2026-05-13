#include "../kernels.h"

void mul_fp32_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
)
{
    float *o = (float *)out;
    const float *x = (const float *)a;
    const float *y = (const float *)b;

    for (int64_t i = 0; i < n; i++)
        o[i] = x[i] * y[i];
}
