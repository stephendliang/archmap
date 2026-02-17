/*
 * Adaptive PF_DIST hyperparameter sweep.
 *
 * Dimensions:
 *   - Counter decay: hard reset (0), >>1, >>2
 *   - Window size: 256, 512, 1024, 2048
 *   - Mutation threshold (raw counter value at decision point)
 *   - PF pair: (PF_LO, PF_HI)
 *
 * Build: gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *            -o test_pf_sweep test_pf_sweep.c -lm
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "simd_map64.h"

/* ---- RNG (deterministic) ---- */
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

/* ---- Zipf ---- */
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

/* ---- workload (pre-generated, reusable) ---- */
struct workload {
    uint64_t *pool;
    uint64_t *op_keys;
    uint8_t  *op_type;
    uint64_t pool_size;
    uint64_t n_ops;
    uint32_t init_live;  /* how many keys are live at start of mixed phase */
};

static struct workload gen_workload(uint64_t pool_size, uint64_t n_ops,
                                     double zipf_s, int pct_lookup, int pct_insert) {
    struct workload w;
    w.pool_size = pool_size;
    w.n_ops = n_ops;
    w.pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    w.op_keys = (uint64_t *)malloc(n_ops * sizeof(uint64_t));
    w.op_type = (uint8_t *)malloc(n_ops);

    struct simd_map64 m; simd_map64_init(&m);
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 1;
        if (simd_map64_insert(&m, k)) w.pool[gen++] = k;
    }
    simd_map64_destroy(&m);

    uint32_t live = (uint32_t)(pool_size / 2), total = (uint32_t)pool_size;
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

/* ---- fixed PF baseline ---- */
static double run_fixed(struct workload *w, int pf_dist) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

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

/* ---- adaptive PF with configurable decay ---- */
static double run_adaptive(struct workload *w,
                            int pf_lo, int pf_hi,
                            int win_bits,     /* window = 1 << win_bits */
                            int shift,        /* 0=hard reset, 1..3 = >>shift */
                            uint32_t thresh)  /* raw counter threshold */
{
    struct simd_map64 m; simd_map64_init(&m);
    for (uint32_t i = 0; i < w->init_live; i++)
        simd_map64_insert(&m, w->pool[i]);

    uint32_t win_mask = (1u << win_bits) - 1;
    uint32_t mut_count = 0;
    int pf_dist = pf_hi;

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        mut_count += (w->op_type[i] != 0);
        if (__builtin_expect((i & win_mask) == 0 && i > 0, 0)) {
            pf_dist = (mut_count > thresh) ? pf_lo : pf_hi;
            if (shift == 0)
                mut_count = 0;
            else
                mut_count >>= shift;
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

/* ---- pure-delete with adaptive load-factor PF ---- */
static double run_delete_adaptive(struct workload *w,
                                   int pf_lo, int pf_hi,
                                   int win_bits,
                                   int load_num, int load_den)  /* threshold: count*den > cap*num */
{
    struct simd_map64 m; simd_map64_init(&m);
    /* insert all pool keys */
    for (uint64_t i = 0; i < w->pool_size; i++)
        simd_map64_insert(&m, w->pool[i]);

    uint32_t win_mask = (1u << win_bits) - 1;
    int pf_dist = pf_hi;

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->pool_size; i++) {
        if (__builtin_expect((i & win_mask) == 0 && i > 0, 0))
            pf_dist = ((uint64_t)m.count * load_den > (uint64_t)m.cap * load_num) ? pf_hi : pf_lo;
        if (i + pf_dist < w->pool_size)
            simd_map64_prefetch2(&m, w->pool[i + pf_dist]);
        tot += (uint64_t)simd_map64_delete(&m, w->pool[i]);
    }
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->pool_size / elapsed / 1e6;
}

static double run_delete_fixed(struct workload *w, int pf_dist) {
    struct simd_map64 m; simd_map64_init(&m);
    for (uint64_t i = 0; i < w->pool_size; i++)
        simd_map64_insert(&m, w->pool[i]);

    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->pool_size; i++) {
        if (i + pf_dist < w->pool_size)
            simd_map64_prefetch2(&m, w->pool[i + pf_dist]);
        tot += (uint64_t)simd_map64_delete(&m, w->pool[i]);
    }
    double elapsed = now_sec() - t0;
    simd_map64_destroy(&m);
    return (double)w->pool_size / elapsed / 1e6;
}

int main(void) {
    uint64_t pool_size = 1000000;
    uint64_t n_ops     = 5000000;
    double   zipf_s    = 1.0;

    struct { const char *name; int lkp, ins; } profiles[] = {
        { "read-heavy",  90,  5 },
        { "balanced",    50, 25 },
        { "churn",       33, 33 },
        { "write-heavy", 10, 50 },
        { "eviction",    20, 10 },
    };
    int n_profiles = 5;

    printf("Adaptive PF_DIST hyperparameter sweep\n");
    printf("pool=%luK  ops=%luM  zipf=%.1f\n", pool_size/1000, n_ops/1000000, zipf_s);
    printf("================================================\n\n");

    /* ---- PART 1: Fixed PF baselines ---- */
    printf("=== Fixed PF baselines ===\n");
    printf("%-12s  PF=4   PF=6   PF=8  PF=10  PF=12  PF=14\n", "profile");
    printf("-------------------------------------------------------\n");
    int fixed_pfs[] = {4, 6, 8, 10, 12, 14};
    int n_fixed = 6;

    for (int p = 0; p < n_profiles; p++) {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, zipf_s,
                                          profiles[p].lkp, profiles[p].ins);
        printf("%-12s", profiles[p].name);
        for (int f = 0; f < n_fixed; f++)
            printf(" %5.1f", run_fixed(&w, fixed_pfs[f]));
        printf("\n");
        free_workload(&w);
    }

    /* pure delete baselines */
    printf("\n%-12s", "pure-del");
    {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, zipf_s, 50, 25);
        for (int f = 0; f < n_fixed; f++)
            printf(" %5.1f", run_delete_fixed(&w, fixed_pfs[f]));
        printf("\n");
        free_workload(&w);
    }

    /* ---- PART 2: Sweep decay + window + threshold for mixed ---- */
    printf("\n=== Adaptive PF sweep (PF_LO=4, PF_HI=12) ===\n");
    printf("decay: 0=hard reset, 1=>>1, 2=>>2\n\n");

    int win_bits_list[] = {8, 9, 10, 11};   /* 256, 512, 1024, 2048 */
    int n_wins = 4;
    int shifts[] = {0, 1, 2};
    int n_shifts = 3;

    /* For each (decay, window), sweep threshold.
     * At steady state with decay >>s, counter before decision ≈ f*WIN * 2^s/(2^s-1)
     * For s=0 (reset): counter = f*WIN.   50% thresh = WIN/2
     * For s=1 (>>1):   counter = f*WIN*2. 50% thresh = WIN
     * For s=2 (>>2):   counter ≈ f*WIN*4/3. 50% thresh ≈ WIN*2/3
     * We sweep fractions: 30%, 40%, 50%, 60%, 70% of the theoretical max
     */
    double fracs[] = {0.30, 0.40, 0.50, 0.60, 0.70};
    int n_fracs = 5;

    /* steady-state multiplier for counter value (before decision) at 100% mutation rate */
    /* shift=0: WIN, shift=1: 2*WIN, shift=2: 4/3*WIN, shift=3: 8/7*WIN */
    double ss_mult[] = {1.0, 2.0, 4.0/3.0};

    for (int p = 0; p < n_profiles; p++) {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, zipf_s,
                                          profiles[p].lkp, profiles[p].ins);
        printf("--- %s (%d/%d/%d) ---\n", profiles[p].name,
               profiles[p].lkp, profiles[p].ins, 100 - profiles[p].lkp - profiles[p].ins);
        printf("%-6s %-5s %-5s", "decay", "win", "thr%");
        printf("  thval  Mops/s\n");

        double best_mops = 0;
        int best_shift = 0, best_win = 0;
        uint32_t best_thresh = 0;
        double best_frac = 0;

        for (int s = 0; s < n_shifts; s++) {
            for (int wb = 0; wb < n_wins; wb++) {
                int win = 1 << win_bits_list[wb];
                for (int fi = 0; fi < n_fracs; fi++) {
                    uint32_t thresh = (uint32_t)(fracs[fi] * ss_mult[shifts[s]] * win);
                    if (thresh < 1) thresh = 1;

                    double mops = run_adaptive(&w, 4, 12,
                                                win_bits_list[wb], shifts[s], thresh);
                    if (mops > best_mops) {
                        best_mops = mops;
                        best_shift = shifts[s];
                        best_win = win;
                        best_thresh = thresh;
                        best_frac = fracs[fi];
                    }
                    printf("  >>%-2d %5d  %3.0f%%  %5u  %5.1f\n",
                           shifts[s], win, fracs[fi]*100, thresh, mops);
                }
            }
        }
        printf("  BEST: decay=>>%d win=%d thresh=%u (%.0f%%) → %.1f Mops/s\n\n",
               best_shift, best_win, best_thresh, best_frac*100, best_mops);
        free_workload(&w);
    }

    /* ---- PART 3: Sweep PF pair values ---- */
    printf("\n=== PF pair sweep (win=1024, decay=>>1 and reset) ===\n");
    int pf_pairs[][2] = {{4,12}, {4,10}, {4,14}, {6,12}, {6,14}, {3,12}, {3,14}};
    int n_pairs = 7;

    for (int p = 0; p < n_profiles; p++) {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, zipf_s,
                                          profiles[p].lkp, profiles[p].ins);
        printf("%-12s", profiles[p].name);
        for (int pi = 0; pi < n_pairs; pi++) {
            /* Use decay=0, thresh=512 (current) for pair comparison */
            double mops = run_adaptive(&w, pf_pairs[pi][0], pf_pairs[pi][1],
                                        10, 0, 512);
            printf("  %d/%d:%5.1f", pf_pairs[pi][0], pf_pairs[pi][1], mops);
        }
        printf("\n");
        free_workload(&w);
    }

    /* ---- PART 4: Pure delete load-factor thresholds ---- */
    printf("\n=== Pure delete: load-factor thresholds ===\n");
    printf("%-8s  fixed4 fixed12  LF>25%%  LF>33%%  LF>50%%  LF>67%%\n", "win");
    {
        seed_rng();
        struct workload w = gen_workload(pool_size, n_ops, zipf_s, 50, 25);

        int del_wins[] = {8, 9, 10, 11};  /* 256, 512, 1024, 2048 */
        /* load factor thresholds: num/den such that count*den > cap*num */
        struct { int num, den; const char *label; } lf_thresh[] = {
            {1, 4, "25%"},  /* count*4 > cap*1 → count > cap/4 → LF>25% */
            {1, 3, "33%"},  /* count*3 > cap*1 → LF>33% */
            {1, 2, "50%"},  /* count*2 > cap*1 → LF>50% (current) */
            {2, 3, "67%"},  /* count*3 > cap*2 → LF>67% */
        };
        int n_lf = 4;

        for (int wb = 0; wb < 4; wb++) {
            int win = 1 << del_wins[wb];
            printf("w=%-5d  %5.1f  %5.1f", win,
                   run_delete_fixed(&w, 4),
                   run_delete_fixed(&w, 12));
            for (int li = 0; li < n_lf; li++) {
                double mops = run_delete_adaptive(&w, 4, 12,
                                                   del_wins[wb],
                                                   lf_thresh[li].num,
                                                   lf_thresh[li].den);
                printf("  %5.1f", mops);
            }
            printf("\n");
        }
        free_workload(&w);
    }

    printf("\nDone.\n");
    return 0;
}
