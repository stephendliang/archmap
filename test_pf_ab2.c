/*
 * Order-controlled A/B: run each variant in isolation with fresh cache.
 * Each variant does its own full insert → delete/mixed cycle.
 * Run interleaved: ABCDE, ABCDE, ABCDE for 3 runs.
 *
 * Build: gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *            -o test_pf_ab2 test_pf_ab2.c -lm
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "simd_map64.h"

static uint64_t rng_s[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t xoshiro256ss(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return result;
}
static void seed_rng(void) {
    uint64_t s = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double *zipf_cdf; static uint64_t zipf_n;
static void zipf_setup(uint64_t n, double s) {
    zipf_n = n; zipf_cdf = malloc(n * sizeof(double));
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) { sum += 1.0/pow((double)(i+1),s); zipf_cdf[i]=sum; }
    for (uint64_t i = 0; i < n; i++) zipf_cdf[i] /= sum;
}
static inline uint64_t zipf_sample(void) {
    double u = (xoshiro256ss() >> 11) * 0x1.0p-53;
    uint64_t lo = 0, hi = zipf_n - 1;
    while (lo < hi) { uint64_t m = lo+(hi-lo)/2; if (zipf_cdf[m]<u) lo=m+1; else hi=m; }
    return lo + 1;
}

/* Generate workload once, reuse across variants */
struct workload {
    uint64_t *pool, *op_keys;
    uint8_t *op_type;
    uint64_t pool_size, n_ops;
    uint32_t init_live;
};

static struct workload gen_workload(uint64_t pool_size, uint64_t n_ops,
                                     double zipf_s, int pct_lookup, int pct_insert) {
    struct workload w;
    w.pool_size = pool_size; w.n_ops = n_ops;
    w.pool = malloc(pool_size * sizeof(uint64_t));
    w.op_keys = malloc(n_ops * sizeof(uint64_t));
    w.op_type = malloc(n_ops);
    struct simd_map64 m; simd_map64_init(&m);
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 1;
        if (simd_map64_insert(&m, k)) w.pool[gen++] = k;
    }
    simd_map64_destroy(&m);
    uint32_t live = pool_size / 2, total = pool_size;
    w.init_live = live;
    int thresh_ins = pct_lookup + pct_insert;
    zipf_setup(live, zipf_s);
    for (uint64_t i = 0; i < n_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
        if (pct < (uint32_t)pct_lookup && live > 0) {
            w.op_type[i] = 0;
            w.op_keys[i] = w.pool[(zipf_sample() - 1) % live];
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            w.op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
            w.op_keys[i] = w.pool[di];
            uint64_t tmp = w.pool[di]; w.pool[di] = w.pool[live]; w.pool[live] = tmp;
            live++;
        } else if (live > 2) {
            w.op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
            w.op_keys[i] = w.pool[li]; live--;
            uint64_t tmp = w.pool[li]; w.pool[li] = w.pool[live]; w.pool[live] = tmp;
        } else {
            w.op_type[i] = 0;
            w.op_keys[i] = (live > 0) ? w.pool[(uint32_t)(xoshiro256ss()>>32)%live] : (xoshiro256ss()|1);
        }
    }
    free(zipf_cdf); zipf_cdf = NULL;
    return w;
}

/* ---- Mixed execution: each variant is a self-contained function ---- */

/* Common setup: insert init_live keys, return map */
static struct simd_map64 setup_map(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);
    return m;
}

/* V0: Fixed PF=12, no adaptation */
__attribute__((noinline))
static double v_fixed12(struct workload *w) {
    struct simd_map64 m = setup_map(w);
    volatile uint64_t tot = 0;
    uint64_t n = w->n_ops;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + 12 < n) {
            if (w->op_type[i + 12] == 0)
                simd_map64_prefetch(&m, w->op_keys[i + 12]);
            else
                simd_map64_prefetch2(&m, w->op_keys[i + 12]);
        }
        switch (w->op_type[i]) {
            case 0: tot += (uint64_t)simd_map64_contains(&m, w->op_keys[i]); break;
            case 1: tot += (uint64_t)simd_map64_insert(&m, w->op_keys[i]); break;
            case 2: tot += (uint64_t)simd_map64_delete(&m, w->op_keys[i]); break;
        }
    }
    double e = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)n / e / 1e6;
}

/* V1: Fixed PF=4 */
__attribute__((noinline))
static double v_fixed4(struct workload *w) {
    struct simd_map64 m = setup_map(w);
    volatile uint64_t tot = 0;
    uint64_t n = w->n_ops;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + 4 < n) {
            if (w->op_type[i + 4] == 0)
                simd_map64_prefetch(&m, w->op_keys[i + 4]);
            else
                simd_map64_prefetch2(&m, w->op_keys[i + 4]);
        }
        switch (w->op_type[i]) {
            case 0: tot += (uint64_t)simd_map64_contains(&m, w->op_keys[i]); break;
            case 1: tot += (uint64_t)simd_map64_insert(&m, w->op_keys[i]); break;
            case 2: tot += (uint64_t)simd_map64_delete(&m, w->op_keys[i]); break;
        }
    }
    double e = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)n / e / 1e6;
}

/* V2: Adaptive, hard reset, win=1024, thresh=512, PF={4,12} (current) */
__attribute__((noinline))
static double v_adapt_reset(struct workload *w) {
    struct simd_map64 m = setup_map(w);
    volatile uint64_t tot = 0;
    uint64_t n = w->n_ops;
    uint32_t mut_count = 0;
    int pf_dist = 12;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > 512) ? 4 : 12;
            mut_count = 0;
        }
        if (i + pf_dist < n) {
            if (w->op_type[i + pf_dist] == 0)
                simd_map64_prefetch(&m, w->op_keys[i + pf_dist]);
            else
                simd_map64_prefetch2(&m, w->op_keys[i + pf_dist]);
        }
        switch (w->op_type[i]) {
            case 0: tot += (uint64_t)simd_map64_contains(&m, w->op_keys[i]); break;
            case 1: tot += (uint64_t)simd_map64_insert(&m, w->op_keys[i]); break;
            case 2: tot += (uint64_t)simd_map64_delete(&m, w->op_keys[i]); break;
        }
    }
    double e = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)n / e / 1e6;
}

/* V3: Adaptive, >>1 decay, win=1024, thresh=1024 */
__attribute__((noinline))
static double v_adapt_decay1(struct workload *w) {
    struct simd_map64 m = setup_map(w);
    volatile uint64_t tot = 0;
    uint64_t n = w->n_ops;
    uint32_t mut_count = 0;
    int pf_dist = 12;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > 1024) ? 4 : 12;
            mut_count >>= 1;
        }
        if (i + pf_dist < n) {
            if (w->op_type[i + pf_dist] == 0)
                simd_map64_prefetch(&m, w->op_keys[i + pf_dist]);
            else
                simd_map64_prefetch2(&m, w->op_keys[i + pf_dist]);
        }
        switch (w->op_type[i]) {
            case 0: tot += (uint64_t)simd_map64_contains(&m, w->op_keys[i]); break;
            case 1: tot += (uint64_t)simd_map64_insert(&m, w->op_keys[i]); break;
            case 2: tot += (uint64_t)simd_map64_delete(&m, w->op_keys[i]); break;
        }
    }
    double e = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)n / e / 1e6;
}

int main(void) {
    uint64_t pool = 1000000, ops = 10000000;

    struct { const char *name; int lkp, ins; } profiles[] = {
        { "read-heavy",  90,  5 },
        { "balanced",    50, 25 },
        { "churn",       33, 33 },
        { "write-heavy", 10, 50 },
        { "eviction",    20, 10 },
    };

    printf("Order-controlled A/B (noinline, 4 runs interleaved)\n");
    printf("V0=fixed12  V1=fixed4  V2=reset  V3=>>1decay\n");
    printf("===================================================\n\n");

    for (int p = 0; p < 5; p++) {
        seed_rng();
        struct workload w = gen_workload(pool, ops, 1.0,
                                          profiles[p].lkp, profiles[p].ins);
        printf("%-12s  ", profiles[p].name);
        double s0=0,s1=0,s2=0,s3=0;
        int n_runs = 4;
        for (int r = 0; r < n_runs; r++) {
            /* rotate starting variant each run to eliminate ordering bias */
            double v[4];
            int order[4];
            for (int k = 0; k < 4; k++) order[k] = (k + r) % 4;
            for (int k = 0; k < 4; k++) {
                switch (order[k]) {
                    case 0: v[0] = v_fixed12(&w); break;
                    case 1: v[1] = v_fixed4(&w); break;
                    case 2: v[2] = v_adapt_reset(&w); break;
                    case 3: v[3] = v_adapt_decay1(&w); break;
                }
            }
            s0 += v[0]; s1 += v[1]; s2 += v[2]; s3 += v[3];
        }
        printf("f12=%5.1f  f4=%5.1f  rst=%5.1f  >>1=%5.1f\n",
               s0/n_runs, s1/n_runs, s2/n_runs, s3/n_runs);
        free(w.pool); free(w.op_keys); free(w.op_type);
    }

    /* Pure delete */
    printf("\npure-delete:  ");
    {
        seed_rng();
        struct workload w = gen_workload(pool, ops, 1.0, 50, 25);
        double sf12=0, slfa=0, slfb=0;
        for (int r = 0; r < 4; r++) {
            /* fixed12 */
            {
                struct simd_map64 m; simd_map64_init(&m);
                for (uint64_t i = 0; i < pool; i++) simd_map64_insert(&m, w.pool[i]);
                seed_rng();
                for (uint64_t i = pool-1; i > 0; i--) {
                    uint64_t j = xoshiro256ss() % (i+1);
                    uint64_t tmp = w.pool[i]; w.pool[i] = w.pool[j]; w.pool[j] = tmp;
                }
                volatile uint64_t tot = 0;
                double t0 = now_sec();
                for (uint64_t i = 0; i < pool; i++) {
                    if (i + 12 < pool) simd_map64_prefetch2(&m, w.pool[i + 12]);
                    tot += (uint64_t)simd_map64_delete(&m, w.pool[i]);
                }
                sf12 += (double)pool / (now_sec() - t0) / 1e6;
                simd_map64_destroy(&m);
            }
            /* LF>50% */
            {
                struct simd_map64 m; simd_map64_init(&m);
                for (uint64_t i = 0; i < pool; i++) simd_map64_insert(&m, w.pool[i]);
                seed_rng();
                for (uint64_t i = pool-1; i > 0; i--) {
                    uint64_t j = xoshiro256ss() % (i+1);
                    uint64_t tmp = w.pool[i]; w.pool[i] = w.pool[j]; w.pool[j] = tmp;
                }
                volatile uint64_t tot = 0; int pf = 12;
                double t0 = now_sec();
                for (uint64_t i = 0; i < pool; i++) {
                    if (__builtin_expect((i & 1023) == 0 && i > 0, 0))
                        pf = (m.count * 2 > m.cap) ? 12 : 4;
                    if (i + pf < pool) simd_map64_prefetch2(&m, w.pool[i + pf]);
                    tot += (uint64_t)simd_map64_delete(&m, w.pool[i]);
                }
                slfa += (double)pool / (now_sec() - t0) / 1e6;
                simd_map64_destroy(&m);
            }
            /* LF>25% */
            {
                struct simd_map64 m; simd_map64_init(&m);
                for (uint64_t i = 0; i < pool; i++) simd_map64_insert(&m, w.pool[i]);
                seed_rng();
                for (uint64_t i = pool-1; i > 0; i--) {
                    uint64_t j = xoshiro256ss() % (i+1);
                    uint64_t tmp = w.pool[i]; w.pool[i] = w.pool[j]; w.pool[j] = tmp;
                }
                volatile uint64_t tot = 0; int pf = 12;
                double t0 = now_sec();
                for (uint64_t i = 0; i < pool; i++) {
                    if (__builtin_expect((i & 1023) == 0 && i > 0, 0))
                        pf = (m.count * 4 > m.cap) ? 12 : 4;
                    if (i + pf < pool) simd_map64_prefetch2(&m, w.pool[i + pf]);
                    tot += (uint64_t)simd_map64_delete(&m, w.pool[i]);
                }
                slfb += (double)pool / (now_sec() - t0) / 1e6;
                simd_map64_destroy(&m);
            }
        }
        printf("f12=%5.1f  LF50=%5.1f  LF25=%5.1f\n",
               sf12/4, slfa/4, slfb/4);
        free(w.pool); free(w.op_keys); free(w.op_type);
    }

    printf("\nDone.\n");
    return 0;
}
