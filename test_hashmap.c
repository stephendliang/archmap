/*
 * C benchmark functions for SAHA tier-1 and verstable.
 * Compiled as C, linked from test_hashmap.cpp.
 */
#include "avx_map64s.h"
#include "avx_map64.h"

#include "testground/khashl/khashl.h"

KHASHL_SET_INIT(static, kh_u64_t, kh_u64, uint64_t, kh_hash_uint64, kh_eq_generic)

#define NAME vt_u64
#define KEY_TY uint64_t
#define HASH_FN vt_hash_integer
#define CMPR_FN vt_cmpr_integer
#include "verstable.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define PF_DIST 8

/* ================================================================
 * xoshiro256** PRNG (fixed seed for reproducibility)
 * ================================================================ */

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}

void bench_seed_rng(void) {
    uint64_t s = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

static inline double rng_uniform(void) {
    return (xoshiro256ss() >> 11) * 0x1.0p-53;
}

/* ================================================================
 * Zipf sampler (inverse-CDF with binary search)
 * ================================================================ */

static double *zipf_cdf;
static uint64_t zipf_n;

static void zipf_setup(uint64_t n, double s) {
    zipf_n = n;
    zipf_cdf = (double *)malloc(n * sizeof(double));
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += 1.0 / pow((double)(i + 1), s);
        zipf_cdf[i] = sum;
    }
    for (uint64_t i = 0; i < n; i++)
        zipf_cdf[i] /= sum;
}

static inline uint64_t zipf_sample(void) {
    double u = rng_uniform();
    uint64_t lo = 0, hi = zipf_n - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (zipf_cdf[mid] < u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo + 1;
}

uint64_t *bench_gen_zipf_keys(uint64_t n, double s, uint64_t n_ops) {
    free(zipf_cdf);
    zipf_setup(n, s);
    uint64_t *keys = (uint64_t *)malloc(n_ops * sizeof(uint64_t));
    for (uint64_t i = 0; i < n_ops; i++)
        keys[i] = zipf_sample();
    free(zipf_cdf);
    zipf_cdf = NULL;
    return keys;
}

/* ================================================================
 * Timing
 * ================================================================ */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * Result struct (must match declaration in .cpp)
 * ================================================================ */

struct bench_result {
    double ins_mops;
    double pos_mops;
    double mix_mops;
    uint64_t unique;
    double dup_pct;
    double hit_pct;
};

/* ================================================================
 * avx_map64 benchmark
 * ================================================================ */

struct bench_result bench_avx64(const uint64_t *k_ins, const uint64_t *k_pos,
                                const uint64_t *k_mix, uint64_t n_ops) {
    struct bench_result r;
    struct avx_map64 m;
    avx_map64_init(&m);

    uint64_t dups = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (avx_map64_insert(&m, k_ins[i]) == 0)
            dups++;
    }
    double elapsed = now_sec() - t0;
    r.ins_mops = (double)n_ops / elapsed / 1e6;
    r.unique   = m.count;
    r.dup_pct  = 100.0 * (double)dups / (double)n_ops;

    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            avx_map64_prefetch(&m, k_pos[i + PF_DIST]);
        avx_map64_contains(&m, k_pos[i]);
    }
    elapsed = now_sec() - t0;
    r.pos_mops = (double)n_ops / elapsed / 1e6;

    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            avx_map64_prefetch(&m, k_mix[i + PF_DIST]);
        if (avx_map64_contains(&m, k_mix[i]))
            hits++;
    }
    elapsed = now_sec() - t0;
    r.mix_mops = (double)n_ops / elapsed / 1e6;
    r.hit_pct  = 100.0 * (double)hits / (double)n_ops;

    avx_map64_destroy(&m);
    return r;
}

/* ================================================================
 * avx_map64s benchmark
 * ================================================================ */

struct bench_result bench_avx64s(const uint64_t *k_ins, const uint64_t *k_pos,
                                 const uint64_t *k_mix, uint64_t n_ops) {
    struct bench_result r;
    struct avx_map64s m;
    avx_map64s_init(&m);

    uint64_t dups = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (avx_map64s_insert(&m, k_ins[i]) == 0)
            dups++;
    }
    double elapsed = now_sec() - t0;
    r.ins_mops = (double)n_ops / elapsed / 1e6;
    r.unique   = m.count;
    r.dup_pct  = 100.0 * (double)dups / (double)n_ops;

    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            avx_map64s_prefetch(&m, k_pos[i + PF_DIST]);
        avx_map64s_contains(&m, k_pos[i]);
    }
    elapsed = now_sec() - t0;
    r.pos_mops = (double)n_ops / elapsed / 1e6;

    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            avx_map64s_prefetch(&m, k_mix[i + PF_DIST]);
        if (avx_map64s_contains(&m, k_mix[i]))
            hits++;
    }
    elapsed = now_sec() - t0;
    r.mix_mops = (double)n_ops / elapsed / 1e6;
    r.hit_pct  = 100.0 * (double)hits / (double)n_ops;

    avx_map64s_destroy(&m);
    return r;
}

/* ================================================================
 * verstable benchmark
 * ================================================================ */

struct bench_result bench_vt(const uint64_t *k_ins, const uint64_t *k_pos,
                             const uint64_t *k_mix, uint64_t n_ops) {
    struct bench_result r;
    vt_u64 set;
    vt_u64_init(&set);

    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++)
        vt_u64_insert(&set, k_ins[i]);
    double elapsed = now_sec() - t0;
    r.ins_mops = (double)n_ops / elapsed / 1e6;
    r.unique   = vt_u64_size(&set);
    r.dup_pct  = 0;

    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++)
        vt_u64_get(&set, k_pos[i]);
    elapsed = now_sec() - t0;
    r.pos_mops = (double)n_ops / elapsed / 1e6;

    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (!vt_u64_is_end(vt_u64_get(&set, k_mix[i])))
            hits++;
    }
    elapsed = now_sec() - t0;
    r.mix_mops = (double)n_ops / elapsed / 1e6;
    r.hit_pct  = 100.0 * (double)hits / (double)n_ops;

    vt_u64_cleanup(&set);
    return r;
}

/* ================================================================
 * khashl benchmark
 * ================================================================ */

struct bench_result bench_khashl(const uint64_t *k_ins, const uint64_t *k_pos,
                                 const uint64_t *k_mix, uint64_t n_ops) {
    struct bench_result r;
    kh_u64_t *h = kh_u64_init();

    uint64_t dups = 0;
    int absent;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        kh_u64_put(h, k_ins[i], &absent);
        if (!absent) dups++;
    }
    double elapsed = now_sec() - t0;
    r.ins_mops = (double)n_ops / elapsed / 1e6;
    r.unique   = kh_size(h);
    r.dup_pct  = 100.0 * (double)dups / (double)n_ops;

    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++)
        kh_u64_get(h, k_pos[i]);
    elapsed = now_sec() - t0;
    r.pos_mops = (double)n_ops / elapsed / 1e6;

    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (kh_u64_get(h, k_mix[i]) != kh_end(h))
            hits++;
    }
    elapsed = now_sec() - t0;
    r.mix_mops = (double)n_ops / elapsed / 1e6;
    r.hit_pct  = 100.0 * (double)hits / (double)n_ops;

    kh_u64_destroy(h);
    return r;
}
