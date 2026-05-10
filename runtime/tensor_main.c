#include <stdio.h>
#include <stdint.h>

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
    print_meta("tensor_1d fp32 [7]", t1);

    int64_t r = 3, c = 4;
    struct base_tensor *t2 = create_tensor_2d(ctx, TYPE_I32, &r, &c);
    failures += expect_ll("2d elems", (long long)t2->num_of_elem, 12);
    failures += expect_sz("2d bytes", t2->num_of_bytes, 12u * 4u);
    print_meta("tensor_2d i32 [3,4]", t2);

    int64_t d0 = 2, d1 = 3, d2 = 5;
    struct base_tensor *t3 =
        create_tensor_3d(ctx, TYPE_FP16, &d0, &d1, &d2);
    failures += expect_ll("3d elems", (long long)t3->num_of_elem, 30);
    failures += expect_sz("3d bytes", t3->num_of_bytes, 30u * 2u);
    print_meta("tensor_3d fp16 [2,3,5]", t3);

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
    print_meta("tensor_nd TYPE_F64 [2,2,2,3]", t5);

    struct base_tensor *mat = create_matrix(ctx, TYPE_FP32, 10, 20);
    failures += expect_ll("matrix elems", (long long)mat->num_of_elem, 200);
    failures += expect_sz("matrix bytes", mat->num_of_bytes, 200u * 4u);
    print_meta("matrix fp32 [10,20]", mat);

    const int64_t pairs[][2] = { { 2, 8 }, { 5, 1 }, { 4, 2 }, { 8, 6 } };
    struct base_tensor *tp =
        create_tensor_from_dim_pairs(ctx, TYPE_FP32, pairs, 4);
    /* 2*8*5*1*4*2*8*6 */
    failures += expect_ll("dim_pairs elems", (long long)tp->num_of_elem, 30720);
    failures += expect_sz("dim_pairs bytes", tp->num_of_bytes, 30720u * 4u);
    print_meta("tensor from dim_pairs -> rank 8", tp);

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
