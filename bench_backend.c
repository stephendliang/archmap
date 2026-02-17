/*
 * bench_backend.c — AVX2 vs AVX-512 backend comparison for simd_map64
 *
 * Self-contained: same PRNG, Zipf sampler, and workloads as test_hashmap.c
 * but depends only on simd_map64.h (no avx_map64s, no boost, no C++).
 *
 * Build two binaries from the same source:
 *   make bench_512    (AVX-512 backend)
 *   make bench_avx2   (AVX2 backend, -mno-avx512f)
 * Run both, diff output.
 *
 * Usage: ./bench_512 [N] [n_ops] [zipf_s]
 */
#include "simd_map64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>

/* ================================================================
 * Backend tag (compile-time)
 * ================================================================ */

#if defined(__AVX512F__)
#define BACKEND_NAME "AVX-512"
#else
#define BACKEND_NAME "AVX2"
#endif

/* ================================================================
 * Hugepage mmap allocator
 * ================================================================ */

#define HP_ROUND(sz) (((sz) + (2u<<20) - 1) & ~((size_t)(2u<<20) - 1))

static void *hp_alloc(size_t bytes) {
    size_t total = HP_ROUND(bytes);
    void *p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                   -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED) madvise(p, total, MADV_HUGEPAGE);
    }
    return p;
}

static void hp_free(void *p, size_t bytes) {
    if (p) munmap(p, HP_ROUND(bytes));
}

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
    zipf_cdf = (double *)hp_alloc(n * sizeof(double));
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

static uint64_t *gen_zipf_keys(uint64_t n, double s, uint64_t n_ops) {
    hp_free(zipf_cdf, zipf_n * sizeof(double));
    zipf_setup(n, s);
    uint64_t *keys = (uint64_t *)hp_alloc(n_ops * sizeof(uint64_t));
    for (uint64_t i = 0; i < n_ops; i++)
        keys[i] = zipf_sample();
    hp_free(zipf_cdf, zipf_n * sizeof(double));
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
 * Prefetch distances (same as test_hashmap.c)
 * ================================================================ */

#define PF_DIST     12
#define PF_DIST_MIX  4

/* ================================================================
 * Insert / lookup+ / lookup± benchmark
 * ================================================================ */

static void bench_core(const uint64_t *k_ins, const uint64_t *k_pos,
                       const uint64_t *k_mix, uint64_t n_ops) {
    struct simd_map64 m;
    simd_map64_init(&m);

    /* insert */
    uint64_t dups = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (simd_map64_insert(&m, k_ins[i]) == 0)
            dups++;
    }
    double elapsed = now_sec() - t0;
    double ins_mops = (double)n_ops / elapsed / 1e6;

    /* lookup+ (positive — all keys exist) */
    t0 = now_sec();
    volatile int sink = 0;
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            simd_map64_prefetch(&m, k_pos[i + PF_DIST]);
        sink += simd_map64_contains(&m, k_pos[i]);
    }
    elapsed = now_sec() - t0;
    double pos_mops = (double)n_ops / elapsed / 1e6;

    /* lookup± (mixed hit/miss) */
    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (i + PF_DIST < n_ops)
            simd_map64_prefetch(&m, k_mix[i + PF_DIST]);
        if (simd_map64_contains(&m, k_mix[i]))
            hits++;
    }
    elapsed = now_sec() - t0;
    double mix_mops = (double)n_ops / elapsed / 1e6;

    printf("  insert:   %6.1f Mops/s  (%lu unique, %.1f%% dup)\n",
           ins_mops, (unsigned long)m.count,
           100.0 * (double)dups / (double)n_ops);
    printf("  lookup+:  %6.1f Mops/s\n", pos_mops);
    printf("  lookup±:  %6.1f Mops/s  (%.1f%% hit)\n",
           mix_mops, 100.0 * (double)hits / (double)n_ops);

    simd_map64_destroy(&m);
}

/* ================================================================
 * Pure delete + mixed workload benchmark
 * ================================================================ */

static void bench_del_mixed(uint64_t pool_size, uint64_t n_mixed_ops,
                            double zipf_s,
                            const char *label, int pct_lookup,
                            int pct_insert, int pct_delete,
                            int print_del) {
    int thresh_ins = pct_lookup + pct_insert;
    (void)pct_delete;

    /* generate pool of unique non-zero keys */
    uint64_t *pool = (uint64_t *)hp_alloc(pool_size * sizeof(uint64_t));
    struct simd_map64 m;
    simd_map64_init(&m);

    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 1;
        if (simd_map64_insert(&m, k))
            pool[gen++] = k;
    }

    /* shuffle for delete order */
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }

    /* pure delete */
    if (print_del) {
        uint64_t tot = 0;
        double t0 = now_sec();
        for (uint64_t i = 0; i < pool_size; i++) {
            if (i + PF_DIST < pool_size)
                simd_map64_prefetch2(&m, pool[i + PF_DIST]);
            tot += (uint64_t)simd_map64_delete(&m, pool[i]);
        }
        double elapsed = now_sec() - t0;
        printf("  delete:   %6.1f Mops/s  (pool=%lu)\n",
               (double)pool_size / elapsed / 1e6,
               (unsigned long)pool_size);
        if (m.count != 0 || tot != pool_size) {
            fprintf(stderr, "  FAIL: delete-all count=%u deleted=%lu\n",
                    m.count, (unsigned long)tot);
        }
        simd_map64_destroy(&m);
    } else {
        simd_map64_destroy(&m);
    }

    /* mixed workload */
    simd_map64_init(&m);
    uint32_t live  = (uint32_t)(pool_size / 2);
    uint32_t total = (uint32_t)pool_size;
    for (uint32_t i = 0; i < live; i++)
        simd_map64_insert(&m, pool[i]);

    zipf_setup(live, zipf_s);

    uint64_t *op_keys = (uint64_t *)hp_alloc(n_mixed_ops * sizeof(uint64_t));
    uint8_t  *op_type = (uint8_t *)hp_alloc(n_mixed_ops);
    uint64_t n_lookups = 0, n_inserts = 0, n_deletes = 0;

    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;

        if (pct < (uint32_t)pct_lookup && live > 0) {
            op_type[i] = 0;
            op_keys[i] = pool[(zipf_sample() - 1) % live];
            n_lookups++;
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
            op_keys[i] = pool[di];
            uint64_t tmp = pool[di]; pool[di] = pool[live]; pool[live] = tmp;
            live++;
            n_inserts++;
        } else if (live > 2) {
            op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
            op_keys[i] = pool[li];
            live--;
            uint64_t tmp = pool[li]; pool[li] = pool[live]; pool[live] = tmp;
            n_deletes++;
        } else {
            op_type[i] = 0;
            op_keys[i] = (live > 0) ? pool[(uint32_t)(xoshiro256ss() >> 32) % live]
                                     : (xoshiro256ss() | 1);
            n_lookups++;
        }
    }

    hp_free(zipf_cdf, zipf_n * sizeof(double));
    zipf_cdf = NULL;

    /* timed section — unified dispatch */
    volatile uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        if (i + PF_DIST_MIX < n_mixed_ops)
            simd_map64_prefetch2(&m, op_keys[i + PF_DIST_MIX]);
        tot += (uint64_t)simd_map64_op(&m, op_keys[i], op_type[i]);
    }
    double elapsed = now_sec() - t0;
    double mixed_mops = (double)n_mixed_ops / elapsed / 1e6;

    /* verify */
    int ok = 1;
    if (m.count != live) {
        fprintf(stderr, "  FAIL: %s count=%u expected=%u\n", label, m.count, live);
        ok = 0;
    }
    if (ok) {
        for (uint32_t i = 0; i < live; i++) {
            if (!simd_map64_contains(&m, pool[i])) {
                fprintf(stderr, "  FAIL: %s live pool[%u] missing\n", label, i);
                ok = 0;
                break;
            }
        }
    }
    if (ok && total > live) {
        for (uint32_t i = live; i < total; i++) {
            if (simd_map64_contains(&m, pool[i])) {
                fprintf(stderr, "  FAIL: %s dead pool[%u] found\n", label, i);
                ok = 0;
                break;
            }
        }
    }

    printf("    %-12s %2d/%2d/%2d:  %6.1f Mops/s   verify: %s\n",
           label, pct_lookup, pct_insert, pct_delete,
           mixed_mops, ok ? "OK" : "FAIL");

    hp_free(op_keys, n_mixed_ops * sizeof(uint64_t));
    hp_free(op_type, n_mixed_ops);
    simd_map64_destroy(&m);
    hp_free(pool, pool_size * sizeof(uint64_t));
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    uint64_t N     = 1000000;
    uint64_t n_ops = 10000000;
    double   zipf_s = 1.0;

    if (argc > 1) N     = (uint64_t)atol(argv[1]);
    if (argc > 2) n_ops = (uint64_t)atol(argv[2]);
    if (argc > 3) zipf_s = atof(argv[3]);

    printf("simd_map64 backend: %s\n", BACKEND_NAME);
    printf("N=%lu  ops=%lu  zipf_s=%.2f\n\n",
           (unsigned long)N, (unsigned long)n_ops, zipf_s);

    seed_rng();

    uint64_t *k_ins = gen_zipf_keys(N, zipf_s, n_ops);
    uint64_t *k_pos = gen_zipf_keys(N, zipf_s, n_ops);
    uint64_t *k_mix = gen_zipf_keys(2 * N, zipf_s, n_ops);

    bench_core(k_ins, k_pos, k_mix, n_ops);

    struct { const char *name; int lkp, ins, del; } profiles[] = {
        { "read-heavy",  90,  5,  5 },
        { "balanced",    50, 25, 25 },
        { "churn",       33, 33, 34 },
        { "write-heavy", 10, 50, 40 },
        { "eviction",    20, 10, 70 },
    };
    int n_profiles = (int)(sizeof(profiles) / sizeof(profiles[0]));

    /* first profile also prints pure-delete */
    printf("  mixed workloads (z=%.1f):\n", zipf_s);
    bench_del_mixed(N, n_ops, zipf_s,
                    profiles[0].name, profiles[0].lkp,
                    profiles[0].ins, profiles[0].del, 1);

    for (int p = 1; p < n_profiles; p++) {
        bench_del_mixed(N, n_ops, zipf_s,
                        profiles[p].name, profiles[p].lkp,
                        profiles[p].ins, profiles[p].del, 0);
    }

    hp_free(k_ins, n_ops * sizeof(uint64_t));
    hp_free(k_pos, n_ops * sizeof(uint64_t));
    hp_free(k_mix, n_ops * sizeof(uint64_t));
    return 0;
}
