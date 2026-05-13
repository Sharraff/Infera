/*
 * IEEE-754 binary16 (fp16) and bfloat16 conversions in software.
 * FP16: round-to-nearest-even on float -> half where practical.
 * BF16: float upper 16 bits; store uses RTNE on the discarded half.
 */
#ifndef HALF_BF16_CONVERT_H
#define HALF_BF16_CONVERT_H

#include <stdint.h>

static inline float infera_bf16_bits_to_f32(uint16_t b)
{
    union {
        float f;
        uint32_t u;
    } x;
    x.u = (uint32_t)b << 16;
    return x.f;
}

static inline uint16_t infera_f32_to_bf16_bits(float f)
{
    union {
        float fl;
        uint32_t u;
    } x;
    x.fl = f;
    /* Round-to-nearest on the low 16 bits of the float32 payload */
    uint32_t u = x.u;
    uint32_t rnd = u + 0x8000u;
    return (uint16_t)(rnd >> 16);
}

static inline float infera_f16_bits_to_f32(uint16_t h)
{
    const uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    const unsigned exp = (h >> 10) & 0x1fu;
    const unsigned mant = h & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = s;
        } else {
            unsigned m = mant;
            unsigned e = 1;
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                e--;
            }
            m &= 0x3ffu;
            f = s | (((uint32_t)(127 - 15 + 1) - e) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        f = s | 0x7f800000u | ((uint32_t)mant << 13);
    } else {
        f = s | (((uint32_t)exp - 15u + 127u) << 23) | ((uint32_t)mant << 13);
    }

    union {
        uint32_t u;
        float fl;
    } o;
    o.u = f;
    return o.fl;
}

static inline uint16_t infera_f32_to_f16_bits(float x)
{
    union {
        float fl;
        uint32_t u;
    } v;
    v.fl = x;
    uint32_t u = v.u;
    uint32_t s = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t m = u & 0x7fffffu;

    if (e <= 0) {
        if (e < -10)
            return (uint16_t)s;
        m |= 0x800000u;
        int32_t t = 14 - e;
        if (t > 31)
            return (uint16_t)s;
        uint32_t outm = m >> (uint32_t)t;
        return (uint16_t)(s | outm);
    }
    if (e >= 31)
        return (uint16_t)(s | 0x7c00u | (m ? 0x200u : 0u));

    uint32_t halfm = m >> 13;
    uint32_t rem = m & 0x1fffu;
    /* RTNE on bit 12 */
    if (rem > 0x1000u || (rem == 0x1000u && (halfm & 1u)))
        halfm++;
    if (halfm >= 0x400u) {
        halfm = 0;
        e++;
        if (e >= 31)
            return (uint16_t)(s | 0x7c00u);
    }
    return (uint16_t)(s | ((uint32_t)e << 10) | halfm);
}

#endif
