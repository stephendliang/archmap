/*
 * bench_del_pf: Compare raw delete vs prefetch-pipelined delete throughput.
 * Proves that the bottleneck is memory latency, not instruction cost.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "avx_map128s.h"

#define N       2000000
#define PF      8
#define ITERS   5

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t *klo, *khi;

static void fill(struct avx_map128s *m) {
    avx_map128s_init(m);
    for (int i = 0; i < N; i++)
        avx_map128s_insert(m, klo[i], khi[i]);
}

int main(void) {
    uint64_t s1 = 0xdeadbeefcafe1234ULL, s2 = 0x0123456789abcdefULL;
    klo = malloc(N * sizeof(uint64_t));
    khi = malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        klo[i] = splitmix64(&s1);
        khi[i] = splitmix64(&s2);
    }

    /* --- Raw delete (no prefetch) --- */
    double best_raw = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        fill(&m);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            avx_map128s_delete(&m, klo[i], khi[i]);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / N;
        if (ns < best_raw) best_raw = ns;
        avx_map128s_destroy(&m);
    }

    /* --- Prefetch-pipelined delete --- */
    double best_pf = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        fill(&m);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        /* prime the pipeline */
        for (int i = 0; i < PF && i < N; i++)
            avx_map128s_prefetch(&m, klo[i], khi[i]);
        for (int i = 0; i < N; i++) {
            if (i + PF < N)
                avx_map128s_prefetch(&m, klo[i + PF], khi[i + PF]);
            avx_map128s_delete(&m, klo[i], khi[i]);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / N;
        if (ns < best_pf) best_pf = ns;
        avx_map128s_destroy(&m);
    }

    /* --- Prefetch-pipelined contains-hit (reference) --- */
    double best_ch = 1e18;
    for (int it = 0; it < ITERS; it++) {
        struct avx_map128s m;
        fill(&m);
        struct timespec t0, t1;
        int sink = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < PF && i < N; i++)
            avx_map128s_prefetch(&m, klo[i], khi[i]);
        for (int i = 0; i < N; i++) {
            if (i + PF < N)
                avx_map128s_prefetch(&m, klo[i + PF], khi[i + PF]);
            sink += avx_map128s_contains(&m, klo[i], khi[i]);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / N;
        if (ns < best_ch) best_ch = ns;
        avx_map128s_destroy(&m);
    }

    printf("delete raw:          %6.1f ns/op\n", best_raw);
    printf("delete prefetched:   %6.1f ns/op\n", best_pf);
    printf("contains prefetched: %6.1f ns/op\n", best_ch);
    printf("speedup:             %.2fx\n", best_raw / best_pf);

    free(klo); free(khi);
    return 0;
}
