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

    int ok = (m.count == N) && dup_ok && (miss == 0) && (false_pos == 0) && (partial_fp == 0);

    /* --- Delete hit: delete first N/2, verify misses, verify remainder --- */
    int del_hit_ok = 1;
    for (int i = 0; i < N / 2; i++) {
        int r = avx_map128s_delete(&m, keys_lo[i], keys_hi[i]);
        if (r != 1) { del_hit_ok = 0; break; }
    }
    if (m.count != (uint32_t)(N - N / 2)) del_hit_ok = 0;
    /* deleted keys must miss */
    for (int i = 0; i < N / 2; i++) {
        if (avx_map128s_contains(&m, keys_lo[i], keys_hi[i])) {
            del_hit_ok = 0; break;
        }
    }
    /* remaining keys must still hit */
    for (int i = N / 2; i < N; i++) {
        if (!avx_map128s_contains(&m, keys_lo[i], keys_hi[i])) {
            del_hit_ok = 0; break;
        }
    }
    printf("  delete hit:   %s (count=%u)\n",
           del_hit_ok ? "PASS" : "FAIL", m.count);
    ok = ok && del_hit_ok;

    /* --- Delete miss: deleting non-existent keys returns 0 --- */
    int del_miss_ok = 1;
    uint64_t dm_seed_lo = 0xAAAABBBBCCCCDDDDULL;
    uint64_t dm_seed_hi = 0x1111222233334444ULL;
    for (int i = 0; i < 1000; i++) {
        uint64_t mlo = splitmix64(&dm_seed_lo);
        uint64_t mhi = splitmix64(&dm_seed_hi);
        if (avx_map128s_delete(&m, mlo, mhi) != 0) {
            del_miss_ok = 0; break;
        }
    }
    printf("  delete miss:  %s\n", del_miss_ok ? "PASS" : "FAIL");
    ok = ok && del_miss_ok;

    /* --- Re-insert after delete: deleted slots are reusable --- */
    int reins_ok = 1;
    for (int i = 0; i < N / 2; i++) {
        int r = avx_map128s_insert(&m, keys_lo[i], keys_hi[i]);
        if (r != 1) { reins_ok = 0; break; }
    }
    for (int i = 0; i < N / 2; i++) {
        if (!avx_map128s_contains(&m, keys_lo[i], keys_hi[i])) {
            reins_ok = 0; break;
        }
    }
    if (m.count != (uint32_t)N) reins_ok = 0;
    printf("  re-insert:    %s (count=%u)\n",
           reins_ok ? "PASS" : "FAIL", m.count);
    ok = ok && reins_ok;

    /* --- Delete all: delete every key, verify count==0 and all miss --- */
    int del_all_ok = 1;
    for (int i = 0; i < N; i++) {
        int r = avx_map128s_delete(&m, keys_lo[i], keys_hi[i]);
        if (r != 1) { del_all_ok = 0; break; }
    }
    if (m.count != 0) del_all_ok = 0;
    for (int i = 0; i < N; i++) {
        if (avx_map128s_contains(&m, keys_lo[i], keys_hi[i])) {
            del_all_ok = 0; break;
        }
    }
    printf("  delete all:   %s (count=%u)\n",
           del_all_ok ? "PASS" : "FAIL", m.count);
    ok = ok && del_all_ok;

    avx_map128s_destroy(&m);

    /* --- init_cap: pre-allocate, insert, verify --- */
    struct avx_map128s m2;
    avx_map128s_init_cap(&m2, N);
    int cap_ok = 1;
    /* Must have enough capacity for N keys without grow */
    if (m2.cap == 0 || m2.data == NULL) cap_ok = 0;
    uint32_t cap_before = m2.cap;
    for (int i = 0; i < N; i++)
        avx_map128s_insert(&m2, keys_lo[i], keys_hi[i]);
    /* Cap should not have changed (no grow needed) */
    if (m2.cap != cap_before) cap_ok = 0;
    if (m2.count != (uint32_t)N) cap_ok = 0;
    /* Spot-check containment */
    for (int i = 0; i < 1000; i++) {
        if (!avx_map128s_contains(&m2, keys_lo[i], keys_hi[i])) {
            cap_ok = 0; break;
        }
    }
    printf("  init_cap:     %s (cap stable=%s, count=%u)\n",
           cap_ok ? "PASS" : "FAIL",
           m2.cap == cap_before ? "yes" : "NO", m2.count);
    ok = ok && cap_ok;
    avx_map128s_destroy(&m2);

    /* --- insert_unique: bulk load without dup check --- */
    struct avx_map128s m3;
    avx_map128s_init_cap(&m3, N);
    int uniq_ok = 1;
    for (int i = 0; i < N; i++)
        avx_map128s_insert_unique(&m3, keys_lo[i], keys_hi[i]);
    if (m3.count != (uint32_t)N) uniq_ok = 0;
    for (int i = 0; i < 1000; i++) {
        if (!avx_map128s_contains(&m3, keys_lo[i], keys_hi[i])) {
            uniq_ok = 0; break;
        }
    }
    printf("  insert_unique: %s (count=%u)\n",
           uniq_ok ? "PASS" : "FAIL", m3.count);
    ok = ok && uniq_ok;
    avx_map128s_destroy(&m3);

    free(keys_lo);
    free(keys_hi);

    return ok ? 0 : 1;
}
