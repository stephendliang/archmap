/*
 * PF_DIST sweep: find optimal prefetch distance for mixed workloads.
 * Also tests batch-pipeline approach (burst prefetch + execute).
 *
 * Build: gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *            -o test_pfdist test_pfdist.c -lm
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>

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

/* ---- workload generation ---- */
struct workload {
    uint64_t *op_keys;
    uint8_t  *op_type;
    uint64_t n_ops;
};

static struct workload gen_workload(uint64_t pool_size, uint64_t n_ops, double zipf_s,
                                    int pct_lookup, int pct_insert) {
    struct workload w;
    w.n_ops = n_ops;
    w.op_keys = (uint64_t *)malloc(n_ops * sizeof(uint64_t));
    w.op_type = (uint8_t *)malloc(n_ops);

    struct simd_map64 m; simd_map64_init(&m);
    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 2;
        if (simd_map64_insert(&m, k)) pool[gen++] = k;
    }
    simd_map64_destroy(&m);

    uint32_t live = (uint32_t)(pool_size / 2), total = (uint32_t)pool_size;
    int thresh_ins = pct_lookup + pct_insert;
    zipf_setup(live, zipf_s);

    for (uint64_t i = 0; i < n_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
        if (pct < (uint32_t)pct_lookup && live > 0) {
            w.op_type[i] = 0; w.op_keys[i] = pool[(zipf_sample()-1)%live];
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            w.op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss()>>32) % (total-live);
            w.op_keys[i] = pool[di];
            uint64_t tmp=pool[di]; pool[di]=pool[live]; pool[live]=tmp; live++;
        } else if (live > 2) {
            w.op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss()>>32) % live;
            w.op_keys[i] = pool[li]; live--;
            uint64_t tmp=pool[li]; pool[li]=pool[live]; pool[live]=tmp;
        } else {
            w.op_type[i] = 0;
            w.op_keys[i] = (live>0) ? pool[(uint32_t)(xoshiro256ss()>>32)%live] : (xoshiro256ss()|2);
        }
    }
    free(zipf_cdf); zipf_cdf = NULL;
    free(pool);
    return w;
}

/* ---- benchmark with variable PF_DIST ---- */
static double bench_pfdist(struct workload *w, int pf_dist, uint64_t pool_size) {
    struct simd_map64 m; simd_map64_init(&m);
    /* pre-fill to 50% of pool */
    seed_rng();
    uint64_t gen = 0;
    while (gen < pool_size / 2) {
        uint64_t k = xoshiro256ss() | 2;
        if (simd_map64_insert(&m, k)) gen++;
    }

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
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

/* ---- dual-distance prefetch: short for lookups, long for mutations ---- */
static double bench_dual_pf(struct workload *w, int pf_short, int pf_long, uint64_t pool_size) {
    struct simd_map64 m; simd_map64_init(&m);
    seed_rng();
    uint64_t gen = 0;
    while (gen < pool_size / 2) {
        uint64_t k = xoshiro256ss() | 2;
        if (simd_map64_insert(&m, k)) gen++;
    }

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        /* short-distance prefetch for lookups */
        if (i + pf_short < w->n_ops && w->op_type[i + pf_short] == 0)
            simd_map64_prefetch(&m, w->op_keys[i + pf_short]);
        /* long-distance prefetch for mutations */
        if (i + pf_long < w->n_ops && w->op_type[i + pf_long] != 0)
            simd_map64_prefetch2(&m, w->op_keys[i + pf_long]);

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

int main(void) {
    uint64_t pool_size = 1000000;
    uint64_t n_ops = 5000000;
    int pf_dists[] = {4, 6, 8, 10, 12, 14, 16, 20, 24, 32};
    int n_pfd = sizeof(pf_dists)/sizeof(pf_dists[0]);

    struct { const char *name; int lkp, ins, del; } profiles[] = {
        { "read-heavy",  90,  5,  5 },
        { "balanced",    50, 25, 25 },
        { "churn",       33, 33, 34 },
        { "write-heavy", 10, 50, 40 },
        { "eviction",    20, 10, 70 },
    };
    int n_profiles = 5;

    printf("PF_DIST sweep (pool=%luK, ops=%luM)\n", pool_size/1000, n_ops/1000000);
    printf("========================================\n\n");

    /* header */
    printf("%-12s", "profile");
    for (int d = 0; d < n_pfd; d++) printf("  PF=%2d", pf_dists[d]);
    printf("   dual\n");
    for (int i = 0; i < 12 + n_pfd * 7 + 7; i++) putchar('-');
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, 0.8,
                                         profiles[p].lkp, profiles[p].ins);
        printf("%-12s", profiles[p].name);
        double best = 0; int best_d = 0;
        for (int d = 0; d < n_pfd; d++) {
            double mops = bench_pfdist(&w, pf_dists[d], pool_size);
            printf("  %5.1f", mops);
            if (mops > best) { best = mops; best_d = pf_dists[d]; }
        }
        /* dual-distance: 8 for lookups, 16 for mutations */
        double dual = bench_dual_pf(&w, 8, 16, pool_size);
        printf("  %5.1f", dual);
        printf("  (best PF=%d)\n", best_d);

        free(w.op_keys); free(w.op_type);
    }

    printf("\n");
    return 0;
}
