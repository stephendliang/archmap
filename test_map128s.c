#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "avx_map128s.h"

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main(void) {
    struct avx_map128s m;
    avx_map128s_init(&m);

    /* Generate N distinct 128-bit keys from two independent PRNG streams */
    uint64_t seed_lo = 0xdeadbeefcafe1234ULL;
    uint64_t seed_hi = 0x0123456789abcdefULL;
    uint64_t *keys_lo = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *keys_hi = (uint64_t *)malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        keys_lo[i] = splitmix64(&seed_lo);
        keys_hi[i] = splitmix64(&seed_hi);
    }

    /* Insert all keys */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        int r = avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);
        if (r != 1) {
            printf("FAIL: insert(%d) returned %d, expected 1\n", i, r);
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ins_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / N;

    /* Duplicate rejection */
    int dup_ok = 1;
    for (int i = 0; i < 1000; i++) {
        if (avx_map128s_insert(&m, keys_lo[i], keys_hi[i]) != 0) {
            printf("FAIL: duplicate insert(%d) returned 1\n", i);
            dup_ok = 0;
        }
    }

    /* Contains: all inserted keys must be found */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int miss = 0;
    for (int i = 0; i < N; i++) {
        if (!avx_map128s_contains(&m, keys_lo[i], keys_hi[i]))
            miss++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double hit_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / N;

    /* Contains: non-inserted keys must not be found */
    int false_pos = 0;
    uint64_t miss_seed_lo = 0xAAAABBBBCCCCDDDDULL;
    uint64_t miss_seed_hi = 0x1111222233334444ULL;
    for (int i = 0; i < N; i++) {
        uint64_t mlo = splitmix64(&miss_seed_lo);
        uint64_t mhi = splitmix64(&miss_seed_hi);
        if (avx_map128s_contains(&m, mlo, mhi))
            false_pos++;
    }

    /* Partial key mismatch: same lo different hi, same hi different lo */
    int partial_fp = 0;
    for (int i = 0; i < 1000; i++) {
        if (avx_map128s_contains(&m, keys_lo[i], keys_hi[i] ^ 1))
            partial_fp++;
        if (avx_map128s_contains(&m, keys_lo[i] ^ 1, keys_hi[i]))
            partial_fp++;
    }

    printf("avx_map128s correctness (N=%d):\n", N);
    printf("  insert:       %s\n", (m.count == N) ? "PASS" : "FAIL");
    printf("  duplicates:   %s\n", dup_ok ? "PASS" : "FAIL");
    printf("  contains hit: %s (miss=%d)\n", miss == 0 ? "PASS" : "FAIL", miss);
    printf("  contains neg: %s (false_pos=%d)\n", false_pos == 0 ? "PASS" : "FAIL", false_pos);
    printf("  partial keys: %s (false_pos=%d)\n", partial_fp == 0 ? "PASS" : "FAIL", partial_fp);
    printf("  insert:   %.1f ns/op\n", ins_ns);
    printf("  contains: %.1f ns/op\n", hit_ns);

    avx_map128s_destroy(&m);
    free(keys_lo);
    free(keys_hi);

    int ok = (m.count == N) && dup_ok && (miss == 0) && (false_pos == 0) && (partial_fp == 0);
    return ok ? 0 : 1;
}
