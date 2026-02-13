/*
 * avx_map64: Zero-metadata direct-key AVX-512 hash set for uint64_t
 *
 * Header-only. Keys stored directly in 8-wide groups (one cache line).
 * vpcmpeqq compares all 8 keys at once — zero false positives, no metadata,
 * no scalar verification. Key=0 reserved as empty sentinel.
 */
#ifndef AVX_MAP64_H
#define AVX_MAP64_H

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define AVX64_INIT_CAP  64    /* 8 groups */
#define AVX64_LOAD_NUM  3
#define AVX64_LOAD_DEN  4     /* 75% load factor */

struct avx_map64 {
    uint64_t *keys;     /* aligned_alloc(64, cap * 8), zero = empty */
    uint32_t count;
    uint32_t cap;       /* ng * 8 */
    uint32_t mask;      /* (cap >> 3) - 1, precomputed for group index */
};

/* --- Hash: fast integer mixer (same as verstable/fast-hash) --- */

static inline uint32_t avx64_hash(uint64_t key) {
    return (uint32_t)_mm_crc32_u64(0, key);
}

/* --- Prefetch helper --- */

static inline void avx_map64_prefetch(struct avx_map64 *m, uint64_t key) {
    uint32_t gi = avx64_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

/* --- SIMD helpers --- */

static inline __mmask8 avx64_match(const uint64_t *grp, uint64_t key) {
    __m512i group  = _mm512_load_si512((const __m512i *)grp);
    __m512i needle = _mm512_set1_epi64((long long)key);
    return _mm512_cmpeq_epi64_mask(group, needle);
}

static inline __mmask8 avx64_empty(const uint64_t *grp) {
    __m512i group = _mm512_load_si512((const __m512i *)grp);
    return _mm512_cmpeq_epi64_mask(group, _mm512_setzero_si512());
}

/* --- Alloc / grow --- */

static size_t avx64_mapsize(uint32_t cap) {
    size_t raw = (size_t)cap * sizeof(uint64_t);
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1); /* round to 2MB */
}

static void avx64_alloc(struct avx_map64 *m, uint32_t cap) {
    size_t total = avx64_mapsize(cap);
    m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (m->keys == MAP_FAILED)
        m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
}

static void avx64_grow(struct avx_map64 *m) {
    uint32_t  old_cap  = m->cap;
    uint64_t *old_keys = m->keys;

    avx64_alloc(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t i = 0; i < old_cap; i++) {
        uint64_t key = old_keys[i];
        if (!key) continue;
        uint32_t gi = avx64_hash(key) & mask;
        for (;;) {
            uint64_t *grp = m->keys + (gi << 3);
            __mmask8 em = avx64_empty(grp);
            if (em) {
                grp[__builtin_ctz(em)] = key;
                m->count++;
                break;
            }
            gi = (gi + 1) & mask;
        }
    }
    munmap(old_keys, avx64_mapsize(old_cap));
}

/* --- Public API --- */

static inline void avx_map64_init(struct avx_map64 *m) {
    memset(m, 0, sizeof(*m));
}

static inline void avx_map64_destroy(struct avx_map64 *m) {
    if (m->keys) munmap(m->keys, avx64_mapsize(m->cap));
}

static inline int avx_map64_insert(struct avx_map64 *m, uint64_t key) {
    if (m->cap == 0) avx64_alloc(m, AVX64_INIT_CAP);
    if (m->count * AVX64_LOAD_DEN >= m->cap * AVX64_LOAD_NUM)
        avx64_grow(m);

    uint32_t gi = avx64_hash(key) & m->mask;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        if (avx64_match(grp, key)) return 0;
        __mmask8 em = avx64_empty(grp);
        if (em) {
            grp[__builtin_ctz(em)] = key;
            m->count++;
            return 1;
        }
        gi = (gi + 1) & m->mask;
    }
}

static inline int avx_map64_contains(struct avx_map64 *m, uint64_t key) {
    if (m->cap == 0) return 0;

    uint32_t gi = avx64_hash(key) & m->mask;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        if (avx64_match(grp, key)) return 1;
        if (avx64_empty(grp))      return 0;
        gi = (gi + 1) & m->mask;
    }
}

#endif /* AVX_MAP64_H */
