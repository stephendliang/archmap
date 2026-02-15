/*
 * test_dispatch.c — A/B test for mixed-workload dispatch strategies
 *
 * AMD uProf measured 11.16% branch misprediction in mixed workloads,
 * primarily from the 3-way switch(op_type) dispatch.  This test
 * explores branch elimination strategies:
 *
 *   switch3   — switch(op) { case 0..2 }  (current baseline)
 *   if_likely — if (__builtin_expect(op==0,1)) ... else if ...
 *   goto_lut  — computed goto: goto *table[op]  (indirect jump)
 *   unified   — single probe loop, op-dependent only at terminal points
 *
 * Build: gcc -O3 -march=native -mavx512f -mavx512bw -o test_dispatch test_dispatch.c -lm
 * Run:   taskset -c 0 ./test_dispatch [pool_size] [n_ops] [zipf_s]
 */

#include "avx_map64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define PF_DIST_MIX  4
#define N_VARIANTS   4
#define N_ROUNDS     4

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

/* ================================================================
 * Zipf sampler
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
    double u = (xoshiro256ss() >> 11) * 0x1.0p-53;
    uint64_t lo = 0, hi = zipf_n - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (zipf_cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo + 1;
}

/* ================================================================
 * Timing
 * ================================================================ */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* avx64_backshift_at and avx_map64_op are now in avx_map64.h */

/* ================================================================
 * Dispatch variants (all noinline to prevent cross-contamination)
 * ================================================================ */

typedef struct { double mops; uint64_t tot; } mix_result_t;

/* --- V0: switch3 (current baseline) --- */
__attribute__((noinline))
static mix_result_t run_switch3(struct avx_map64 *m, const uint64_t *keys,
                                 const uint8_t *ops, uint64_t n) {
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + PF_DIST_MIX < n) {
            if (ops[i + PF_DIST_MIX] == 0)
                avx_map64_prefetch(m, keys[i + PF_DIST_MIX]);
            else
                avx_map64_prefetch2(m, keys[i + PF_DIST_MIX]);
        }
        switch (ops[i]) {
            case 0: tot += (uint64_t)avx_map64_contains(m, keys[i]); break;
            case 1: tot += (uint64_t)avx_map64_insert(m, keys[i]); break;
            case 2: tot += (uint64_t)avx_map64_delete(m, keys[i]); break;
        }
    }
    return (mix_result_t){ (double)n / (now_sec() - t0) / 1e6, tot };
}

/* --- V1: if_likely (bias first branch toward contains) --- */
__attribute__((noinline))
static mix_result_t run_if_likely(struct avx_map64 *m, const uint64_t *keys,
                                   const uint8_t *ops, uint64_t n) {
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + PF_DIST_MIX < n) {
            if (ops[i + PF_DIST_MIX] == 0)
                avx_map64_prefetch(m, keys[i + PF_DIST_MIX]);
            else
                avx_map64_prefetch2(m, keys[i + PF_DIST_MIX]);
        }
        if (__builtin_expect(ops[i] == 0, 1)) {
            tot += (uint64_t)avx_map64_contains(m, keys[i]);
        } else if (ops[i] == 1) {
            tot += (uint64_t)avx_map64_insert(m, keys[i]);
        } else {
            tot += (uint64_t)avx_map64_delete(m, keys[i]);
        }
    }
    return (mix_result_t){ (double)n / (now_sec() - t0) / 1e6, tot };
}

/* --- V2: computed goto (indirect jump, no call overhead) --- */
__attribute__((noinline))
static mix_result_t run_goto_lut(struct avx_map64 *m, const uint64_t *keys,
                                  const uint8_t *ops, uint64_t n) {
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + PF_DIST_MIX < n) {
            if (ops[i + PF_DIST_MIX] == 0)
                avx_map64_prefetch(m, keys[i + PF_DIST_MIX]);
            else
                avx_map64_prefetch2(m, keys[i + PF_DIST_MIX]);
        }
        static void *dtable[] = { &&Lcontains, &&Linsert, &&Ldelete };
        goto *dtable[ops[i]];
        Lcontains: tot += (uint64_t)avx_map64_contains(m, keys[i]); goto Ldone;
        Linsert:   tot += (uint64_t)avx_map64_insert(m, keys[i]);   goto Ldone;
        Ldelete:   tot += (uint64_t)avx_map64_delete(m, keys[i]);   goto Ldone;
        Ldone: ;
    }
    return (mix_result_t){ (double)n / (now_sec() - t0) / 1e6, tot };
}

/* --- V3: unified (single probe, late dispatch, always pf2) --- */
__attribute__((noinline))
static mix_result_t run_unified(struct avx_map64 *m, const uint64_t *keys,
                                 const uint8_t *ops, uint64_t n) {
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n; i++) {
        if (i + PF_DIST_MIX < n)
            avx_map64_prefetch2(m, keys[i + PF_DIST_MIX]);
        tot += (uint64_t)avx_map64_op(m, keys[i], ops[i]);
    }
    return (mix_result_t){ (double)n / (now_sec() - t0) / 1e6, tot };
}

/* ================================================================
 * Runner infrastructure
 * ================================================================ */

typedef mix_result_t (*run_fn_t)(struct avx_map64 *, const uint64_t *,
                                  const uint8_t *, uint64_t);

static const char *vnames[N_VARIANTS] = {
    "switch3", "if_likely", "goto_lut", "unified"
};
static run_fn_t runners[N_VARIANTS] = {
    run_switch3, run_if_likely, run_goto_lut, run_unified
};

int main(int argc, char **argv) {
    uint64_t pool_size = argc > 1 ? (uint64_t)atol(argv[1]) : 1000000;
    uint64_t n_ops     = argc > 2 ? (uint64_t)atol(argv[2]) : 10000000;
    double   zipf_s    = argc > 3 ? atof(argv[3]) : 1.0;

    seed_rng();

    /* generate unique non-zero key pool */
    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    {
        struct avx_map64 tmp;
        avx_map64_init(&tmp);
        uint64_t gen = 0;
        while (gen < pool_size) {
            uint64_t k = xoshiro256ss() | 1;
            if (avx_map64_insert(&tmp, k))
                pool[gen++] = k;
        }
        avx_map64_destroy(&tmp);
    }

    /* Fisher-Yates shuffle */
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t t = pool[i]; pool[i] = pool[j]; pool[j] = t;
    }

    struct { const char *name; int lkp, ins, del; } profiles[] = {
        { "read-heavy", 90, 5, 5 },
        { "balanced",   50, 25, 25 },
        { "churn",      33, 33, 33 },
        { "write-heavy", 10, 50, 40 },
        { "eviction",   20, 10, 70 },
    };
    int n_profiles = sizeof(profiles) / sizeof(profiles[0]);

    printf("dispatch A/B: pool=%lu ops=%lu z=%.1f  (%d rounds, order-rotated)\n\n",
           (unsigned long)pool_size, (unsigned long)n_ops, zipf_s, N_ROUNDS);
    printf("%-12s", "profile");
    for (int v = 0; v < N_VARIANTS; v++)
        printf("  %9s", vnames[v]);
    printf("  verify\n");

    for (int p = 0; p < n_profiles; p++) {
        int pct_lkp = profiles[p].lkp;
        int thresh_ins = pct_lkp + profiles[p].ins;

        /* track live/dead pool for op generation */
        uint64_t *pcopy = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
        memcpy(pcopy, pool, pool_size * sizeof(uint64_t));
        uint32_t live = (uint32_t)(pool_size / 2);
        uint32_t total = (uint32_t)pool_size;

        zipf_setup(live, zipf_s);

        uint64_t *op_keys = (uint64_t *)malloc(n_ops * sizeof(uint64_t));
        uint8_t  *op_type = (uint8_t *)malloc(n_ops);

        for (uint64_t i = 0; i < n_ops; i++) {
            uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
            if (pct < (uint32_t)pct_lkp && live > 0) {
                op_type[i] = 0;
                op_keys[i] = pcopy[(zipf_sample() - 1) % live];
            } else if (pct < (uint32_t)thresh_ins && live < total) {
                op_type[i] = 1;
                uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
                op_keys[i] = pcopy[di];
                uint64_t t = pcopy[di]; pcopy[di] = pcopy[live]; pcopy[live] = t;
                live++;
            } else if (live > 2) {
                op_type[i] = 2;
                uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
                op_keys[i] = pcopy[li];
                live--;
                uint64_t t = pcopy[li]; pcopy[li] = pcopy[live]; pcopy[live] = t;
            } else {
                op_type[i] = 0;
                op_keys[i] = (live > 0)
                    ? pcopy[(uint32_t)(xoshiro256ss() >> 32) % live]
                    : (xoshiro256ss() | 1);
            }
        }
        uint32_t expected_count = live;
        uint32_t init_live = (uint32_t)(pool_size / 2);

        free(zipf_cdf); zipf_cdf = NULL;
        free(pcopy);

        /* benchmark with order rotation */
        double results[N_VARIANTS] = {0};
        uint64_t ref_tot = 0;
        uint32_t ref_count = 0;
        int verified = 1;

        for (int round = 0; round < N_ROUNDS; round++) {
            for (int vi = 0; vi < N_VARIANTS; vi++) {
                int v = (vi + round) % N_VARIANTS;

                /* fresh map with initial live pool */
                struct avx_map64 m;
                avx_map64_init(&m);
                for (uint32_t j = 0; j < init_live; j++)
                    avx_map64_insert(&m, pool[j]);

                mix_result_t r = runners[v](&m, op_keys, op_type, n_ops);
                results[v] += r.mops;

                /* verify on round 0 */
                if (round == 0) {
                    if (vi == 0) {
                        ref_tot = r.tot;
                        ref_count = m.count;
                        if (m.count != expected_count) {
                            fprintf(stderr, "FAIL: %s count=%u expected=%u\n",
                                    vnames[v], m.count, expected_count);
                            verified = 0;
                        }
                    } else if (r.tot != ref_tot || m.count != ref_count) {
                        fprintf(stderr, "FAIL: %s tot=%lu count=%u "
                                "(expected %lu/%u)\n",
                                vnames[v], (unsigned long)r.tot, m.count,
                                (unsigned long)ref_tot, ref_count);
                        verified = 0;
                    }
                }

                avx_map64_destroy(&m);
            }
        }

        printf("%-12s", profiles[p].name);
        for (int v = 0; v < N_VARIANTS; v++)
            printf("  %8.1f", results[v] / N_ROUNDS);
        printf("  %s\n", verified ? "OK" : "FAIL");

        free(op_keys);
        free(op_type);
    }

    free(pool);
    return 0;
}
