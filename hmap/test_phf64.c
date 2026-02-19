#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#define PHF64_SLOTS 16
#define AVX_PHF64_PREFETCH
#include "avx_phf64.h"
#include "simd_map64.h"

#define N        2000000
#define PF_AHEAD 24

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double elapsed_ns(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
}

static int test_config(const char *label, uint64_t *keys,
                       uint64_t *miss_keys, uint32_t min_ng) {
    struct avx_phf64 m;
    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    int attempts = avx_phf64_build(&m, keys, N, min_ng);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double build_ms = elapsed_ns(t0, t1) / 1e6;

    if (attempts < 0) {
        printf("  %-12s  FAILED (alloc)\n", label);
        return 0;
    }

    uint32_t n_groups = m.mask + 1;
    uint32_t cap = n_groups << PHF64_SHIFT;

    /* Max group occupancy */
    uint32_t max_occ = 0;
    for (uint32_t gi = 0; gi < n_groups; gi++) {
        uint32_t occ = 0;
        uint64_t *grp = m.keys + (gi << PHF64_SHIFT);
        for (int s = 0; s < PHF64_SLOTS; s++)
            if (grp[s] != 0) occ++;
        if (occ > max_occ) max_occ = occ;
    }

    /* Correctness */
    int miss = 0;
    for (int i = 0; i < N; i++)
        if (!avx_phf64_contains(&m, keys[i])) miss++;

    int false_pos = 0;
    for (int i = 0; i < N; i++)
        if (avx_phf64_contains(&m, miss_keys[i])) false_pos++;

    int ok = (miss == 0) && (false_pos == 0);

    /* Benchmark: raw hit */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile int sink = 0;
    for (int i = 0; i < N; i++)
        sink += avx_phf64_contains(&m, keys[i]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double raw_hit = elapsed_ns(t0, t1) / N;

    /* Benchmark: pipelined hit */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sink = 0;
    for (int i = 0; i < PF_AHEAD && i < N; i++)
        avx_phf64_prefetch(&m, keys[i]);
    for (int i = 0; i < N; i++) {
        if (i + PF_AHEAD < N)
            avx_phf64_prefetch(&m, keys[i + PF_AHEAD]);
        sink += avx_phf64_contains(&m, keys[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double pf_hit = elapsed_ns(t0, t1) / N;

    /* Benchmark: raw miss */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sink = 0;
    for (int i = 0; i < N; i++)
        sink += avx_phf64_contains(&m, miss_keys[i]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double raw_miss = elapsed_ns(t0, t1) / N;

    double mem_mb = phf64_mapsize(cap) / (1024.0 * 1024.0);

    printf("  %-12s %4.1f%%  max=%2u/%2d  seeds=%-4d %6.1fms  %5.0fMB  "
           "hit=%5.1f  pf=%5.1f  miss=%5.1f  %s\n",
           label,
           100.0 * N / cap, max_occ, PHF64_SLOTS,
           attempts, build_ms, mem_mb,
           raw_hit, pf_hit, raw_miss,
           ok ? "PASS" : "FAIL");

    avx_phf64_destroy(&m);
    return ok;
}

int main(void) {
    /* Generate keys */
    uint64_t *keys = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *miss_keys = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t s1 = 0xdeadbeefcafe1234ULL;
    uint64_t s2 = 0xAAAABBBBCCCCDDDDULL;
    for (int i = 0; i < N; i++) {
        uint64_t k;
        do { k = splitmix64(&s1); } while (k == 0);
        keys[i] = k;
        do { k = splitmix64(&s2); } while (k == 0);
        miss_keys[i] = k;
    }

    /* Dynamic avx_map64 baseline */
    struct simd_map64 dyn;
    simd_map64_init(&dyn);
    for (int i = 0; i < N; i++)
        simd_map64_insert(&dyn, keys[i]);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile int sink = 0;
    for (int i = 0; i < N; i++)
        sink += simd_map64_contains(&dyn, keys[i]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dyn_hit = elapsed_ns(t0, t1) / N;
    simd_map64_destroy(&dyn);

    printf("avx_phf64 16-slot (2 CL) sweep (N=%d, PF=%d):\n", N, PF_AHEAD);
    printf("  %-12s %5s  %8s  %5s %7s  %5s  %5s  %5s  %5s  %s\n",
           "config", "load", "max", "seeds", "build", "mem",
           "hit", "pf", "miss", "ok");
    printf("  -----------------------------------------------------------"
           "-------------------------------\n");

    int ok = 1;

    /* 16-slot groups: ceiling is ~24% at ng=n/4 (λ≈3.8) */
    struct { const char *name; uint32_t min_ng; } configs[] = {
        { "ng=n/4",  N / 4  },   /* λ≈3.8, ~24% load — the ceiling */
        { "ng=n/2",  N / 2  },   /* λ≈1.9, ~12% load */
        { "ng=n",    N      },   /* λ≈0.95, ~6% load */
        { "ng=2n",   N * 2  },   /* λ≈0.48, ~3% load */
        { "auto",    0      },
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < n_configs; c++)
        ok &= test_config(configs[c].name, keys, miss_keys, configs[c].min_ng);

    printf("  -----------------------------------------------------------"
           "-------------------------------\n");
    printf("  simd_map64 (dynamic, 8-slot, 75%% load):   "
           "                              hit=%5.1f\n", dyn_hit);

    free(keys);
    free(miss_keys);
    return ok ? 0 : 1;
}
