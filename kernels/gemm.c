#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"


void gemm_8x4_avx2(
    int K,
    const float *A,   // 8 x K
    const float *B,   // K x 4
    float *C,         // 8 x 4
    int ldc
) {
    __m256 c0 = _mm256_loadu_ps(C + 0*ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1*ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3*ldc);

    for (int k = 0; k < K; k++) {
        __m256 a = _mm256_loadu_ps(A + k*8);

        __m256 b0 = _mm256_broadcast_ss(B + k*4 + 0);
        __m256 b1 = _mm256_broadcast_ss(B + k*4 + 1);
        __m256 b2 = _mm256_broadcast_ss(B + k*4 + 2);
        __m256 b3 = _mm256_broadcast_ss(B + k*4 + 3);

        c0 = _mm256_fmadd_ps(a, b0, c0);
        c1 = _mm256_fmadd_ps(a, b1, c1);
        c2 = _mm256_fmadd_ps(a, b2, c2);
        c3 = _mm256_fmadd_ps(a, b3, c3);
    }

    _mm256_storeu_ps(C + 0*ldc, c0);
    _mm256_storeu_ps(C + 1*ldc, c1);
    _mm256_storeu_ps(C + 2*ldc, c2);
    _mm256_storeu_ps(C + 3*ldc, c3);
}

int main() {
    int K = 8;
    float A[8*K];
    float B[K*4];
    float C[8*4];

    // initialize
    for (int i = 0; i < 8*K; i++) A[i] = 1.0f;
    for (int i = 0; i < K*4; i++) B[i] = 1.0f;
    for (int i = 0; i < 8*4; i++) C[i] = 0.0f;

    gemm_8x4_avx2(K, A, B, C, 8);

    // print result
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", C[i + j*8]);
        }
        printf("\n");
    }

    return 0;
}