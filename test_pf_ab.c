/*
 * Focused A/B test: compare adaptive PF variants on official benchmark
 * structure (separate-compilation simulation via noinline).
 *
 * Tests 3 variants × 5 profiles, interleaved to minimize system noise.
 *
 * Build: gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *            -o test_pf_ab test_pf_ab.c -lm
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "simd_map64.h"

/* ---- RNG ---- */
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
    zipf_n = n; zipf_cdf = (double *)malloc(n * sizeof(double));
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

/* ---- Workload ---- */
struct workload {
    uint64_t *pool;
    uint64_t *op_keys;
    uint8_t  *op_type;
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
static void free_workload(struct workload *w) {
    free(w->pool); free(w->op_keys); free(w->op_type);
}

/* ========================================
 * Mixed workload variants (noinline to simulate separate TU)
 * ======================================== */

/* Variant A: Current — hard reset, win=1024, thresh=512, LF>50% for delete */
__attribute__((noinline))
static double run_mixed_A(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    uint32_t mut_count = 0;
    int pf_dist = 12;
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > 512) ? 4 : 12;
            mut_count = 0;
        }
        if (i + pf_dist < w->n_ops) {
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
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->n_ops / elapsed / 1e6;
}

/* Variant B: >>1 decay, win=1024, thresh=1024 (steady-state equiv of 50%) */
__attribute__((noinline))
static double run_mixed_B(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    uint32_t mut_count = 0;
    int pf_dist = 12;
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > 1024) ? 4 : 12;
            mut_count >>= 1;
        }
        if (i + pf_dist < w->n_ops) {
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
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->n_ops / elapsed / 1e6;
}

/* Variant C: >>1 decay, win=512, thresh=512 */
__attribute__((noinline))
static double run_mixed_C(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    uint32_t mut_count = 0;
    int pf_dist = 12;
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & 511) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > 512) ? 4 : 12;
            mut_count >>= 1;
        }
        if (i + pf_dist < w->n_ops) {
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
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->n_ops / elapsed / 1e6;
}

/* Variant D: Fixed PF=12 baseline (no adaptation overhead) */
__attribute__((noinline))
static double run_mixed_fixed12(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + 12 < w->n_ops) {
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
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->n_ops / elapsed / 1e6;
}

/* Variant E: Fixed PF=4 baseline */
__attribute__((noinline))
static double run_mixed_fixed4(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + 4 < w->n_ops) {
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
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->n_ops / elapsed / 1e6;
}

/* ---- Pure delete variants ---- */
__attribute__((noinline))
static double run_del_lf50(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint64_t i = 0; i < w->pool_size; i++)
        simd_map64_insert(&m, w->pool[i]);
    /* shuffle */
    for (uint64_t i = w->pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = w->pool[i]; w->pool[i] = w->pool[j]; w->pool[j] = tmp;
    }
    int pf_dist = 12;
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->pool_size; i++) {
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0))
            pf_dist = (m.count * 2 > m.cap) ? 12 : 4;
        if (i + pf_dist < w->pool_size)
            simd_map64_prefetch2(&m, w->pool[i + pf_dist]);
        tot += (uint64_t)simd_map64_delete(&m, w->pool[i]);
    }
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->pool_size / elapsed / 1e6;
}

__attribute__((noinline))
static double run_del_lf25(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint64_t i = 0; i < w->pool_size; i++)
        simd_map64_insert(&m, w->pool[i]);
    for (uint64_t i = w->pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = w->pool[i]; w->pool[i] = w->pool[j]; w->pool[j] = tmp;
    }
    int pf_dist = 12;
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->pool_size; i++) {
        if (__builtin_expect((i & 1023) == 0 && i > 0, 0))
            pf_dist = (m.count * 4 > m.cap) ? 12 : 4;
        if (i + pf_dist < w->pool_size)
            simd_map64_prefetch2(&m, w->pool[i + pf_dist]);
        tot += (uint64_t)simd_map64_delete(&m, w->pool[i]);
    }
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->pool_size / elapsed / 1e6;
}

__attribute__((noinline))
static double run_del_fixed12(struct workload *w) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint64_t i = 0; i < w->pool_size; i++)
        simd_map64_insert(&m, w->pool[i]);
    for (uint64_t i = w->pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = w->pool[i]; w->pool[i] = w->pool[j]; w->pool[j] = tmp;
    }
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->pool_size; i++) {
        if (i + 12 < w->pool_size)
            simd_map64_prefetch2(&m, w->pool[i + 12]);
        tot += (uint64_t)simd_map64_delete(&m, w->pool[i]);
    }
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->pool_size / elapsed / 1e6;
}

int main(void) {
    uint64_t pool = 1000000, ops = 10000000;
    int n_runs = 3;

    struct { const char *name; int lkp, ins; } profiles[] = {
        { "read-heavy",  90,  5 },
        { "balanced",    50, 25 },
        { "churn",       33, 33 },
        { "write-heavy", 10, 50 },
        { "eviction",    20, 10 },
    };

    printf("A/B test: adaptive PF variants (noinline, pool=1M, ops=10M)\n");
    printf("A = reset/1024/512   B = >>1/1024/1024   C = >>1/512/512\n");
    printf("============================================================\n\n");

    /* ---- Pure delete A/B ---- */
    printf("=== Pure delete (3 interleaved runs) ===\n");
    printf("         fixed12  LF>50%%   LF>25%%\n");
    {
        seed_rng();
        struct workload w = gen_workload(pool, ops, 1.0, 50, 25);
        double f12[3], a50[3], a25[3];
        for (int r = 0; r < n_runs; r++) {
            seed_rng();  /* reset shuffle seed for consistent order */
            f12[r] = run_del_fixed12(&w);
            seed_rng();
            a50[r] = run_del_lf50(&w);
            seed_rng();
            a25[r] = run_del_lf25(&w);
            printf("  run %d:  %5.1f    %5.1f    %5.1f\n", r+1, f12[r], a50[r], a25[r]);
        }
        double avg_f12 = (f12[0]+f12[1]+f12[2])/3;
        double avg_a50 = (a50[0]+a50[1]+a50[2])/3;
        double avg_a25 = (a25[0]+a25[1]+a25[2])/3;
        printf("  avg:    %5.1f    %5.1f    %5.1f\n\n", avg_f12, avg_a50, avg_a25);
        free_workload(&w);
    }

    /* ---- Mixed workload A/B ---- */
    printf("=== Mixed workloads (3 interleaved runs) ===\n");
    printf("%-12s  fixed12  fixed4   A(rst)  B(>>1)  C(>>1/512)\n", "profile");
    printf("-------------------------------------------------------------\n");

    for (int p = 0; p < 5; p++) {
        seed_rng();
        struct workload w = gen_workload(pool, ops, 1.0,
                                          profiles[p].lkp, profiles[p].ins);
        double sum_f12=0, sum_f4=0, sum_a=0, sum_b=0, sum_c=0;
        for (int r = 0; r < n_runs; r++) {
            double mf12 = run_mixed_fixed12(&w);
            double mf4  = run_mixed_fixed4(&w);
            double ma   = run_mixed_A(&w);
            double mb   = run_mixed_B(&w);
            double mc   = run_mixed_C(&w);
            sum_f12 += mf12; sum_f4 += mf4;
            sum_a += ma; sum_b += mb; sum_c += mc;
        }
        printf("%-12s  %5.1f    %5.1f   %5.1f   %5.1f   %5.1f\n",
               profiles[p].name,
               sum_f12/n_runs, sum_f4/n_runs,
               sum_a/n_runs, sum_b/n_runs, sum_c/n_runs);
        free_workload(&w);
    }

    printf("\nDone.\n");
    return 0;
}
