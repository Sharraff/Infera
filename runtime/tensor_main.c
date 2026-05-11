#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

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
    default:
        printf("dtype %d not supported for value print", (int)t->dtype);
        break;
    }

    if (show < n)
        printf(", ...");
    printf("]\n");
}

static int expect_ll(const char *what, long long got, long long want)
{
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %lld want %lld\n", what, got, want);
        return 1;
    }
    return 0;
}

static int expect_sz(const char *what, size_t got, size_t want)
{
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %zu want %zu\n", what, got, want);
        return 1;
    }
    return 0;
}

int main(void)
{
    int failures = 0;
    struct context *ctx = NULL;

    int64_t n = 7;
    struct base_tensor *t1 = create_tensor_1d(ctx, TYPE_FP32, &n);
    failures += expect_ll("1d elems", (long long)t1->num_of_elem, 7);
    failures += expect_sz("1d bytes", t1->num_of_bytes, 7u * 4u);
    fill_tensor_with_demo_values(t1);
    print_meta("tensor_1d fp32 [7]", t1);
    print_tensor_values("tensor_1d fp32 [7]", t1, 16);

    int64_t r = 3, c = 4;
    struct base_tensor *t2 = create_tensor_2d(ctx, TYPE_I32, &r, &c);
    failures += expect_ll("2d elems", (long long)t2->num_of_elem, 12);
    failures += expect_sz("2d bytes", t2->num_of_bytes, 12u * 4u);
    fill_tensor_with_demo_values(t2);
    print_meta("tensor_2d i32 [3,4]", t2);
    print_tensor_values("tensor_2d i32 [3,4]", t2, 16);

    int64_t d0 = 2, d1 = 3, d2 = 5;
    struct base_tensor *t3 =
        create_tensor_3d(ctx, TYPE_FP16, &d0, &d1, &d2);
    failures += expect_ll("3d elems", (long long)t3->num_of_elem, 30);
    failures += expect_sz("3d bytes", t3->num_of_bytes, 30u * 2u);
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
    failures += expect_ll("4d elems", (long long)t5->num_of_elem, 24);
    failures += expect_sz("4d bytes", t5->num_of_bytes, 24u * 8u);
    fill_tensor_with_demo_values(t5);
    print_meta("tensor_nd TYPE_F64 [2,2,2,3]", t5);
    print_tensor_values("tensor_nd TYPE_F64 [2,2,2,3]", t5, 20);

    struct base_tensor *mat = create_matrix(ctx, TYPE_FP32, 10, 20);
    failures += expect_ll("matrix elems", (long long)mat->num_of_elem, 200);
    failures += expect_sz("matrix bytes", mat->num_of_bytes, 200u * 4u);
    fill_tensor_with_demo_values(mat);
    print_meta("matrix fp32 [10,20]", mat);
    print_tensor_values("matrix fp32 [10,20]", mat, 20);

    const int64_t pairs[][2] = { { 2, 8 }, { 5, 1 }, { 4, 2 }, { 8, 6 } };
    struct base_tensor *tp =
        create_tensor_from_dim_pairs(ctx, TYPE_FP32, pairs, 4);
    /* 2*8*5*1*4*2*8*6 */
    failures += expect_ll("dim_pairs elems", (long long)tp->num_of_elem, 30720);
    failures += expect_sz("dim_pairs bytes", tp->num_of_bytes, 30720u * 4u);
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
