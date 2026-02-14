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
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    uint32_t gi = avx64_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle))        return 1;
        if (_mm512_testn_epi64_mask(group, group))          return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Backshift delete: O(1) expected, no tombstones --- */

static inline int avx_map64_delete(struct avx_map64 *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    uint32_t gi   = avx64_hash(key) & m->mask;
    uint32_t mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    /* locate the key — single SIMD load, dual mask extraction */
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm    = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);

        if (mm) {
            grp[__builtin_ctz(mm)] = 0;
            m->count--;

            /* group was non-full before delete → no probe chain to fix */
            if (empty) return 1;

            /* backshift: pull displaced keys toward their home group.
             *
             * Invariant: for key K at group G with home H, groups H..G-1
             * are all completely full.  We just punched a hole in a full
             * group, so any key downstream whose probe path crossed this
             * group is now unreachable via contains (which stops at the
             * first group with an empty slot).  Walk forward and pull
             * one key per group back to fill each successive hole.
             */
            uint32_t hole_gi = gi;
            int hole_slot = __builtin_ctz(mm); /* slot we just zeroed */
            uint32_t scan_gi = (gi + 1) & mask;

            for (;;) {
                /* prefetch 2 groups ahead — sequential scan is predictable */
                uint32_t pf_gi = (scan_gi + 2) & mask;
                _mm_prefetch((const char *)(m->keys + (pf_gi << 3)), _MM_HINT_T0);

                uint64_t *scan_grp = m->keys + (scan_gi << 3);
                __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
                __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

                if (scan_empty == 0xFF) break; /* fully empty group — chain over */

                int moved = 0;
                for (__mmask8 todo = (~scan_empty) & 0xFF; todo; todo &= todo - 1) {
                    int s = __builtin_ctz(todo);
                    uint32_t home = avx64_hash(scan_grp[s]) & mask;
                    /* does this key's probe path cross hole_gi? */
                    if (((hole_gi - home) & mask) < ((scan_gi - home) & mask)) {
                        uint64_t *hole_grp = m->keys + (hole_gi << 3);
                        hole_grp[hole_slot] = scan_grp[s];
                        scan_grp[s] = 0;

                        /* scan group already had empties → chain ends here */
                        if (scan_empty) return 1;

                        hole_gi = scan_gi;
                        hole_slot = s; /* moved key's slot is the new hole */
                        moved = 1;
                        break;
                    }
                }

                if (!moved && scan_empty) break; /* empties, no candidates — done */
                scan_gi = (scan_gi + 1) & mask;
            }
            return 1;
        }
        if (empty) return 0; /* key not found */
        gi = (gi + 1) & mask;
    }
}

#endif /* AVX_MAP64_H */
