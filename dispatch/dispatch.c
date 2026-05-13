#include "dispatch.h"

#include <stddef.h>

#include "../kernels/kernels.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCC__)
#if defined(__has_builtin)
#if __has_builtin(__builtin_cpu_supports)
#define INFERA_USE_RUNTIME_AVX2 1
#endif
#elif defined(__GNUC__) && __GNUC__ >= 6
#define INFERA_USE_RUNTIME_AVX2 1
#endif
#endif

static struct binary_dispatch_table fp32_binary;

static void infera_fp32_dispatch_init(void)
{
#ifdef INFERA_USE_RUNTIME_AVX2
    if (__builtin_cpu_supports("avx2")) {
        fp32_binary.add = add_fp32_kernel_avx2;
        fp32_binary.sub = sub_fp32_kernel_avx2;
        fp32_binary.mul = mul_fp32_kernel_avx2;
        fp32_binary.div = div_fp32_kernel_avx2;
        return;
    }
#endif
    fp32_binary.add = add_fp32_kernel;
    fp32_binary.sub = sub_fp32_kernel;
    fp32_binary.mul = mul_fp32_kernel;
    fp32_binary.div = div_fp32_kernel;
}

static void __attribute__((constructor)) infera_dispatch_init(void)
{
    infera_fp32_dispatch_init();
}

static struct binary_dispatch_table fp16_binary = {
    .add = add_fp16_kernel,
    .sub = sub_fp16_kernel,
    .mul = mul_fp16_kernel,
    .div = div_fp16_kernel,
};

static struct binary_dispatch_table bf16_binary = {
    .add = add_bf16_kernel,
    .sub = sub_bf16_kernel,
    .mul = mul_bf16_kernel,
    .div = div_bf16_kernel,
};

struct binary_dispatch_table *dispatch_binary_tables[DISPATCH_BINARY_SLOTS] = {
    [TYPE_FP32] = &fp32_binary,
    [TYPE_FP16] = &fp16_binary,
    [TYPE_BF16] = &bf16_binary,
};
