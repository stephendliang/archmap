/*
 * bench_del128s: A/B benchmark — old (rehash) vs new (displacement) backshift delete
 *
 * Measures: pure insert, pure contains-hit, pure contains-miss,
 *           pure delete, mixed insert/delete churn, delete+contains interleave.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double elapsed_ns(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
}

/* Pull in the variant selected at compile time */
#ifdef USE_OLD
#include "avx_map128s_old.h"
#define LABEL "OLD (rehash)"
#else
#include "avx_map128s.h"
#define LABEL "NEW (displacement)"
#endif

#define N       2000000
#define WARMUP  200000
#define ITERS   3

static uint64_t *keys_lo, *keys_hi;
static uint64_t *miss_lo, *miss_hi;

static void gen_keys(void) {
    uint64_t s1 = 0xdeadbeefcafe1234ULL;
    uint64_t s2 = 0x0123456789abcdefULL;
    keys_lo = malloc(N * sizeof(uint64_t));
    keys_hi = malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        keys_lo[i] = splitmix64(&s1);
        keys_hi[i] = splitmix64(&s2);
    }
    uint64_t ms1 = 0xAAAABBBBCCCCDDDDULL;
    uint64_t ms2 = 0x1111222233334444ULL;
    miss_lo = malloc(N * sizeof(uint64_t));
    miss_hi = malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        miss_lo[i] = splitmix64(&ms1);
        miss_hi[i] = splitmix64(&ms2);
    }
}

/* --- Benchmark functions --- */

static double bench_insert(void) {
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        avx_map128s_init(&m);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / N;
        if (ns < best) best = ns;
        avx_map128s_destroy(&m);
    }
    return best;
}

static double bench_contains_hit(struct avx_map128s *m) {
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct timespec t0, t1;
        int sink = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            sink += avx_map128s_contains(m, keys_lo[i], keys_hi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / N;
        if (ns < best) best = ns;
        if (sink != N) printf("  BUG: hit sink=%d\n", sink);
    }
    return best;
}

static double bench_contains_miss(struct avx_map128s *m) {
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct timespec t0, t1;
        int sink = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            sink += avx_map128s_contains(m, miss_lo[i], miss_hi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / N;
        if (ns < best) best = ns;
        if (sink != 0) printf("  BUG: miss sink=%d\n", sink);
    }
    return best;
}

static double bench_delete(void) {
    /* insert all, then time deleting all */
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        avx_map128s_init(&m);
        for (int i = 0; i < N; i++)
            avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            avx_map128s_delete(&m, keys_lo[i], keys_hi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / N;
        if (ns < best) best = ns;
        if (m.count != 0) printf("  BUG: count=%u after delete-all\n", m.count);
        avx_map128s_destroy(&m);
    }
    return best;
}

static double bench_churn(void) {
    /* Fill to N, then churn: delete first half, re-insert, repeat */
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        avx_map128s_init(&m);
        for (int i = 0; i < N; i++)
            avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);

        int half = N / 2;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        /* 3 churn rounds */
        for (int round = 0; round < 3; round++) {
            for (int i = 0; i < half; i++)
                avx_map128s_delete(&m, keys_lo[i], keys_hi[i]);
            for (int i = 0; i < half; i++)
                avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / (half * 6); /* 6 ops per round × 3 */
        if (ns < best) best = ns;
        avx_map128s_destroy(&m);
    }
    return best;
}

static double bench_delete_contains(void) {
    /* Delete half, then contains-hit on remaining half */
    double best = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        avx_map128s_init(&m);
        for (int i = 0; i < N; i++)
            avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);

        int half = N / 2;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < half; i++)
            avx_map128s_delete(&m, keys_lo[i], keys_hi[i]);
        int sink = 0;
        for (int i = half; i < N; i++)
            sink += avx_map128s_contains(&m, keys_lo[i], keys_hi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ns(t0, t1) / N;
        if (ns < best) best = ns;
        avx_map128s_destroy(&m);
    }
    return best;
}

int main(void) {
    gen_keys();
    printf("=== %s (N=%d) ===\n", LABEL, N);

    double ins = bench_insert();
    printf("  insert:         %6.1f ns/op\n", ins);

    /* build a map for contains benchmarks */
    struct avx_map128s m;
    avx_map128s_init(&m);
    for (int i = 0; i < N; i++)
        avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);

    double hit = bench_contains_hit(&m);
    printf("  contains-hit:   %6.1f ns/op\n", hit);

    double mis = bench_contains_miss(&m);
    printf("  contains-miss:  %6.1f ns/op\n", mis);

    avx_map128s_destroy(&m);

    double del = bench_delete();
    printf("  delete-all:     %6.1f ns/op\n", del);

    double churn = bench_churn();
    printf("  churn (del+ins):%6.1f ns/op\n", churn);

    double dc = bench_delete_contains();
    printf("  del+contains:   %6.1f ns/op\n", dc);

    free(keys_lo); free(keys_hi);
    free(miss_lo); free(miss_hi);
    return 0;
}
