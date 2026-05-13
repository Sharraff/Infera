#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

void add_fp32_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void sub_fp32_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void mul_fp32_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void div_fp32_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
void add_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void sub_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void mul_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void div_fp32_kernel_avx2(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);
#endif

void add_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void sub_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void mul_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void div_fp16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void add_bf16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void sub_bf16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void mul_bf16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

void div_bf16_kernel(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

#endif
