/*
 * avx_phf64: Perfect hash set for uint64_t using AVX-512
 *
 * Header-only. Static set built once from a known key array.
 * A CRC32 seed is searched at build time so that no group exceeds
 * PHF64_SLOTS keys. Runtime lookup is 1-2 cache-line loads + vpcmpeqq.
 * No probing, no loop, guaranteed O(1) worst case.
 * Key=0 reserved as empty sentinel.
 *
 * Define PHF64_SLOTS before including to set group width:
 *   8  = 1 CL per group, max ~12% slot load
 *   16 = 2 CL per group, max ~24% slot load
 */
#ifndef AVX_PHF64_H
#define AVX_PHF64_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#ifndef PHF64_SLOTS
#define PHF64_SLOTS 8
#endif

#if PHF64_SLOTS == 8
#define PHF64_SHIFT 3
#elif PHF64_SLOTS == 16
#define PHF64_SHIFT 4
#else
#error "PHF64_SLOTS must be 8 or 16"
#endif

struct avx_phf64 {
    uint64_t *keys;     /* 64-byte aligned, zero = empty */
    uint64_t seed;      /* added to key before CRC32 */
    uint32_t mask;      /* n_groups - 1 */
};

/* --- Hash: add seed to key, then CRC32 ---
 * Addition is nonlinear over GF(2), so different seeds produce
 * genuinely different group distributions (unlike XOR, which
 * merely permutes group labels due to CRC32 linearity). */

static inline uint32_t phf64_hash(uint64_t seed, uint64_t key) {
    return (uint32_t)_mm_crc32_u64(0, key + seed);
}

/* --- Allocation (same mmap/hugepage pattern as avx_map64) --- */

static size_t phf64_mapsize(uint32_t cap) {
    size_t raw = (size_t)cap * sizeof(uint64_t);
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static uint64_t *phf64_alloc(uint32_t cap) {
    size_t total = phf64_mapsize(cap);
    uint64_t *p = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                                   | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) {
        p = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                             -1, 0);
        if (p != MAP_FAILED)
            madvise(p, total, MADV_HUGEPAGE);
    }
    return p;
}

/* --- Seed generator (splitmix64) --- */

static inline uint64_t phf64_splitmix(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* --- Build: find seed, place keys ---
 *
 * min_ng: minimum number of groups (0 = auto: next_pow2(n),
 * targeting ~12.5% slot load at 8 slots, ~6% at 16 slots).
 * Build tries seeds; if no seed works within 1000 attempts,
 * doubles groups and retries.
 * Returns seed attempt count (>0) or -1 on alloc failure.
 */
static int avx_phf64_build(struct avx_phf64 *m,
                            const uint64_t *keys, uint32_t n,
                            uint32_t min_ng) {
    uint32_t ng = 1;
    if (min_ng == 0) {
        uint32_t target = n < PHF64_SLOTS ? 1 : n;
        while (ng < target) ng <<= 1;
    } else {
        while (ng < min_ng) ng <<= 1;
    }

    uint32_t *counts = NULL;
    uint32_t cap, mask;
    uint64_t seed;
    uint64_t seed_state = 0;
    int total_attempts = 0;
    for (;;) {
        cap = ng << PHF64_SHIFT;
        mask = ng - 1;
        counts = (uint32_t *)realloc(counts, ng * sizeof(uint32_t));
        if (!counts) return -1;

        int found = 0;
        for (int attempt = 0; attempt < 1000; attempt++) {
            seed = phf64_splitmix(&seed_state);
            total_attempts++;
            memset(counts, 0, ng * sizeof(uint32_t));
            int ok = 1;
            for (uint32_t i = 0; i < n; i++) {
                uint32_t gi = phf64_hash(seed, keys[i]) & mask;
                if (++counts[gi] > PHF64_SLOTS) { ok = 0; break; }
            }
            if (ok) { found = 1; break; }
        }
        if (found) break;
        ng <<= 1; /* double groups and retry */
    }
    free(counts);

    /* Allocate and place keys */
    uint64_t *table = phf64_alloc(cap);
    if (table == MAP_FAILED) return -1;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t gi = phf64_hash(seed, keys[i]) & mask;
        uint64_t *grp = table + (gi << PHF64_SHIFT);
        for (int s = 0; s < PHF64_SLOTS; s++) {
            if (grp[s] == 0) { grp[s] = keys[i]; break; }
        }
    }

    m->keys = table;
    m->seed = seed;
    m->mask = mask;
    return total_attempts;
}

/* --- Runtime: single-group lookup (1 or 2 cache lines) --- */

static inline int avx_phf64_contains(const struct avx_phf64 *m, uint64_t key) {
    uint32_t gi = phf64_hash(m->seed, key) & m->mask;
    uint64_t *grp = m->keys + (gi << PHF64_SHIFT);
    __m512i needle = _mm512_set1_epi64((long long)key);
#if PHF64_SLOTS <= 8
    return !!_mm512_cmpeq_epi64_mask(_mm512_load_si512(grp), needle);
#else
    __mmask8 m0 = _mm512_cmpeq_epi64_mask(_mm512_load_si512(grp), needle);
    __mmask8 m1 = _mm512_cmpeq_epi64_mask(_mm512_load_si512(grp + 8), needle);
    return !!(m0 | m1);
#endif
}

#ifdef AVX_PHF64_PREFETCH
/* --- Prefetch: software-pipeline batch lookups --- */

static inline void avx_phf64_prefetch(const struct avx_phf64 *m, uint64_t key) {
    uint32_t gi = phf64_hash(m->seed, key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << PHF64_SHIFT)), _MM_HINT_T0);
#if PHF64_SLOTS > 8
    _mm_prefetch((const char *)(m->keys + (gi << PHF64_SHIFT) + 8), _MM_HINT_T0);
#endif
}
#endif

/* --- Destroy --- */

static inline void avx_phf64_destroy(struct avx_phf64 *m) {
    if (m->keys)
        munmap(m->keys, phf64_mapsize((m->mask + 1) << PHF64_SHIFT));
}

#endif /* AVX_PHF64_H */
