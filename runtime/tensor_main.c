#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include "internal/half_bf16_convert.h"
#include "tensor_runtime.h"

static void print_meta(const char *label, struct base_tensor *t)
{
    if (!t) {
        printf("%s: (null)\n", label);
        return;
    }
    printf(
        "%s: elements=%lld bytes=%zu dtype=%d data=%p\n",
        label,
        (long long)t->num_of_elem,
        t->num_of_bytes,
        (int)t->dtype,
        (void *)t->data
    );
}

static void fill_tensor_with_demo_values(struct base_tensor *t)
{
    if (!t || !t->data)
        return;

    int64_t n = t->num_of_elem;
    switch (t->dtype) {
    case TYPE_FP32: {
        float *p = (float *)t->data;
        for (int64_t i = 0; i < n; i++)
            p[i] = (float)i + 0.5f;
        break;
    }
    case TYPE_I32: {
        int32_t *p = (int32_t *)t->data;
        for (int64_t i = 0; i < n; i++)
            p[i] = (int32_t)(i * 3);
        break;
    }
    case TYPE_F64: {
        double *p = (double *)t->data;
        for (int64_t i = 0; i < n; i++)
            p[i] = (double)i + 0.25;
        break;
    }
    case TYPE_FP16: {
        /*
         * Runtime stores fp16 payloads as raw 16-bit values. For demo output,
         * write increasing bit patterns and print them as hex words.
         */
        uint16_t *p = (uint16_t *)t->data;
        for (int64_t i = 0; i < n; i++)
            p[i] = (uint16_t)(0x3C00u + (uint16_t)i);
        break;
    }
    default:
        break;
    }
}

static void print_tensor_values(const char *label, struct base_tensor *t, int max_print)
{
    if (!t || !t->data) {
        printf("%s values: (null)\n", label);
        return;
    }

    int64_t n = t->num_of_elem;
    int64_t show = n < max_print ? n : max_print;

    printf("%s values (flat, first %lld/%lld): [",
           label, (long long)show, (long long)n);

    switch (t->dtype) {
    case TYPE_FP32: {
        const float *p = (const float *)t->data;
        for (int64_t i = 0; i < show; i++) {
            if (i)
                printf(", ");
            printf("%.2f", p[i]);
        }
        break;
    }
    case TYPE_I32: {
        const int32_t *p = (const int32_t *)t->data;
        for (int64_t i = 0; i < show; i++) {
            if (i)
                printf(", ");
            printf("%" PRId32, p[i]);
        }
        break;
    }
    case TYPE_F64: {
        const double *p = (const double *)t->data;
        for (int64_t i = 0; i < show; i++) {
            if (i)
                printf(", ");
            printf("%.2f", p[i]);
        }
        break;
    }
    case TYPE_FP16: {
        const uint16_t *p = (const uint16_t *)t->data;
        for (int64_t i = 0; i < show; i++) {
            if (i)
                printf(", ");
            printf("0x%04" PRIx16, p[i]);
        }
        break;
    }
    case TYPE_BF16: {
        const uint16_t *p = (const uint16_t *)t->data;
        for (int64_t i = 0; i < show; i++) {
            if (i)
                printf(", ");
            printf("0x%04" PRIx16, p[i]);
        }
        break;
    }
    default:
        printf("dtype %d not supported for value print", (int)t->dtype);
        break;
    }

    if (show < n)
        printf(", ...");
    printf("]\n");
}

static void print_fp32_matrix_3x3(const char *label, struct base_tensor *t)
{
    if (!t || t->dtype != TYPE_FP32 || !t->data) {
        printf("%s: (invalid tensor)\n", label);
        return;
    }
    if (t->num_dims != 2 || t->shape[0] != 3 || t->shape[1] != 3) {
        printf("%s: expected shape [3,3]\n", label);
        return;
    }
    const float *p = (const float *)t->data;
    printf("%s:\n", label);
    for (int r = 0; r < 3; r++) {
        printf("  [");
        for (int c = 0; c < 3; c++) {
            if (c)
                printf(" ");
            printf("%.2f", p[r * 3 + c]);
        }
        printf("]\n");
    }
}


int main(void)
{
    int failures = 0;
    struct context *ctx = NULL;

    struct base_tensor *m_a = create_matrix(ctx, TYPE_FP32, 3, 3);
    struct base_tensor *m_b = create_matrix(ctx, TYPE_FP32, 3, 3);
    if (!m_a || !m_b) {
        fprintf(stderr, "FAIL: could not allocate 3x3 matrices\n");
        failures++;
    } else {
        static const float vals_a[9] = {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
        };
        static const float vals_b[9] = {
            9.f, 8.f, 7.f,
            6.f, 5.f, 4.f,
            3.f, 2.f, 1.f,
        };
        memcpy(m_a->data, vals_a, sizeof(vals_a));
        memcpy(m_b->data, vals_b, sizeof(vals_b));

        printf("--- fp32 binary ops via dispatch (scalar or AVX2 at runtime), add/sub/mul/div, (3,3) ---\n");
        print_fp32_matrix_3x3("Matrix A", m_a);
        print_fp32_matrix_3x3("Matrix B", m_b);

        struct base_tensor *m_sum = tensor_add(ctx, m_a, m_b);
        if (!m_sum) {
            fprintf(stderr, "FAIL: tensor_add returned NULL\n");
            failures++;
        } else {
            print_fp32_matrix_3x3("A + B", m_sum);
            tensor_destroy(m_sum);
        }

        struct base_tensor *m_diff = tensor_sub(ctx, m_a, m_b);
        if (!m_diff) {
            fprintf(stderr, "FAIL: tensor_sub returned NULL\n");
            failures++;
        } else {
            print_fp32_matrix_3x3("A - B", m_diff);
            tensor_destroy(m_diff);
        }

        struct base_tensor *m_prod = tensor_mul(ctx, m_a, m_b);
        if (!m_prod) {
            fprintf(stderr, "FAIL: tensor_mul returned NULL\n");
            failures++;
        } else {
            print_fp32_matrix_3x3("A * B (elementwise)", m_prod);
            tensor_destroy(m_prod);
        }

        struct base_tensor *m_quot = tensor_div(ctx, m_a, m_b);
        if (!m_quot) {
            fprintf(stderr, "FAIL: tensor_div returned NULL\n");
            failures++;
        } else {
            print_fp32_matrix_3x3("A / B", m_quot);
            tensor_destroy(m_quot);
        }

        tensor_destroy(m_a);
        tensor_destroy(m_b);
    }

    {
        int64_t sz = 2;
        struct base_tensor *h1 = create_tensor_2d(ctx, TYPE_FP16, &sz, &sz);
        struct base_tensor *h2 = create_tensor_2d(ctx, TYPE_FP16, &sz, &sz);
        if (h1 && h2) {
            uint16_t *p1 = (uint16_t *)h1->data;
            uint16_t *p2 = (uint16_t *)h2->data;
            for (int i = 0; i < 4; i++) {
                p1[i] = infera_f32_to_f16_bits((float)(i + 1));
                p2[i] = infera_f32_to_f16_bits(2.f);
            }
            printf("--- fp16 elementwise mul via dispatch, (2,2) ---\n");
            struct base_tensor *hp = tensor_mul(ctx, h1, h2);
            if (!hp) {
                fprintf(stderr, "FAIL: fp16 tensor_mul returned NULL\n");
                failures++;
            } else {
                print_tensor_values("fp16 A*B (raw u16)", hp, 4);
                tensor_destroy(hp);
            }
            tensor_destroy(h1);
            tensor_destroy(h2);
        }
    }

    {
        int64_t sz = 2;
        struct base_tensor *b1 = create_tensor_2d(ctx, TYPE_BF16, &sz, &sz);
        struct base_tensor *b2 = create_tensor_2d(ctx, TYPE_BF16, &sz, &sz);
        if (b1 && b2) {
            uint16_t *p1 = (uint16_t *)b1->data;
            uint16_t *p2 = (uint16_t *)b2->data;
            for (int i = 0; i < 4; i++) {
                p1[i] = infera_f32_to_bf16_bits((float)(i + 1));
                p2[i] = infera_f32_to_bf16_bits(0.5f);
            }
            printf("--- bf16 elementwise add via dispatch, (2,2) ---\n");
            struct base_tensor *bs = tensor_add(ctx, b1, b2);
            if (!bs) {
                fprintf(stderr, "FAIL: bf16 tensor_add returned NULL\n");
                failures++;
            } else {
                print_tensor_values("bf16 A+B (raw u16)", bs, 4);
                tensor_destroy(bs);
            }
            tensor_destroy(b1);
            tensor_destroy(b2);
        }
    }

    int64_t n = 7;
    struct base_tensor *t1 = create_tensor_1d(ctx, TYPE_FP32, &n);
    if (t1 == NULL) {
        fprintf(stderr, "FAIL: create_tensor_1d returned NULL\n");
    }

    fill_tensor_with_demo_values(t1);
    print_meta("tensor_1d fp32 [7]", t1);
    print_tensor_values("tensor_1d fp32 [7]", t1, 16);

    int64_t r = 3, c = 4;
    struct base_tensor *t2 = create_tensor_2d(ctx, TYPE_I32, &r, &c);
    if (t2 == NULL) {
        fprintf(stderr, "FAIL: create_tensor_2d returned NULL\n");
    }
    
    fill_tensor_with_demo_values(t2);
    print_meta("tensor_2d i32 [3,4]", t2);
    print_tensor_values("tensor_2d i32 [3,4]", t2, 16);

    int64_t d0 = 2, d1 = 3, d2 = 5;
    struct base_tensor *t3 =
        create_tensor_3d(ctx, TYPE_FP16, &d0, &d1, &d2);
        if (t3 == NULL) {
            fprintf(stderr, "FAIL: create_tensor_3d returned NULL\n");
        }
    
    fill_tensor_with_demo_values(t3);
    print_meta("tensor_3d fp16 [2,3,5]", t3);
    print_tensor_values("tensor_3d fp16 [2,3,5]", t3, 20);

    int64_t shape[] = { 2, 2, 2, 3 };
    struct base_tensor *t4 =
        create_tensor(ctx, (enum base_type)9999, shape, 4);
    if (t4 != NULL) {
        fprintf(stderr, "FAIL: invalid type should return NULL\n");
        failures++;
    } else {
        printf("create_tensor invalid dtype: correctly returned NULL\n");
    }

    struct base_tensor *t5 =
        create_tensor(ctx, TYPE_F64, shape, 4);
        if (t5 == NULL) {
            fprintf(stderr, "FAIL: create_tensor_5d returned NULL\n");
        }
    
    print_meta("tensor_nd TYPE_F64 [2,2,2,3]", t5);
    print_tensor_values("tensor_nd TYPE_F64 [2,2,2,3]", t5, 20);

    struct base_tensor *mat = create_matrix(ctx, TYPE_FP32, 10, 20);
    if (mat == NULL) {
        fprintf(stderr, "FAIL: failed to create_matrix\n");
    }


    fill_tensor_with_demo_values(mat);
    print_meta("matrix fp32 [10,20]", mat);
    print_tensor_values("matrix fp32 [10,20]", mat, 20);

    const int64_t pairs[][2] = { { 2, 8 }, { 5, 1 }, { 4, 2 }, { 8, 6 } };
    struct base_tensor *tp =
        create_tensor_from_dim_pairs(ctx, TYPE_FP32, pairs, 4);
    if (tp == NULL) {
        fprintf(stderr, "FAIL: failed to create_tensor_from_dim_pairs\n");
    }
    /* 2*8*5*1*4*2*8*6 */
    
    fill_tensor_with_demo_values(tp);
    print_meta("tensor from dim_pairs -> rank 8", tp);
    print_tensor_values("tensor from dim_pairs -> rank 8", tp, 20);

    void *buf = create_buffer(ctx, 128);
    printf("create_buffer(128) -> %p\n", (void *)buf);
    free(buf);

    tensor_destroy(t1);
    tensor_destroy(t2);
    tensor_destroy(t3);
    tensor_destroy(t5);
    tensor_destroy(mat);
    tensor_destroy(tp);

    if (failures == 0) {
        printf("All checks passed.\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}
