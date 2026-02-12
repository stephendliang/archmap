/*
 * avx_map64: Native uint64_t AVX-512 hash set
 *
 * Header-only. Same SIMD probing as SAHA tier-1 but with a fast integer
 * mixer instead of wyhash, and a native uint64_t API (no pack/len overhead).
 */
#ifndef AVX_MAP64_H
#define AVX_MAP64_H

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AVX64_INIT_CAP  64
#define AVX64_LOAD_NUM  7
#define AVX64_LOAD_DEN  8

struct avx_map64 {
    uint8_t  *meta;
    uint64_t *keys;
    uint32_t count;
    uint32_t cap;
};

/* --- Hash: fast integer mixer (same as verstable/fast-hash) --- */

static inline uint64_t avx64_hash(uint64_t key) {
    key ^= key >> 23;
    key *= 0x2127599bf4325c37ULL;
    key ^= key >> 47;
    return key;
}

/* --- SIMD helpers (same logic as SAHA) --- */

static inline uint8_t avx64_h2(uint64_t hash) {
    return (uint8_t)((hash >> 57) | 0x80);
}

static inline __mmask64 avx64_match(const uint8_t *meta, uint8_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi8((char)h2);
    return _mm512_cmpeq_epi8_mask(group, needle);
}

static inline __mmask64 avx64_empty(const uint8_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_cmpeq_epi8_mask(group, _mm512_setzero_si512());
}

/* --- Alloc / grow --- */

static void avx64_alloc(struct avx_map64 *m, uint32_t cap) {
    m->meta = (uint8_t *)aligned_alloc(64, cap);
    memset(m->meta, 0, cap);
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->cap   = cap;
    m->count = 0;
}

static void avx64_grow(struct avx_map64 *m) {
    uint32_t old_cap  = m->cap;
    uint8_t  *old_meta = m->meta;
    uint64_t *old_keys = m->keys;

    avx64_alloc(m, old_cap * 2);
    uint32_t ng = m->cap / 64;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!(old_meta[i] & 0x80)) continue;
        uint64_t key = old_keys[i];
        uint64_t h   = avx64_hash(key);
        uint8_t  h2  = avx64_h2(h);
        uint32_t gi  = (uint32_t)h & (ng - 1);
        for (;;) {
            __mmask64 em = avx64_empty(m->meta + gi * 64);
            if (em) {
                int pos = __builtin_ctzll(em);
                m->meta[gi * 64 + pos] = h2;
                m->keys[gi * 64 + pos] = key;
                m->count++;
                break;
            }
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

/* --- Public API --- */

static inline void avx_map64_init(struct avx_map64 *m) {
    memset(m, 0, sizeof(*m));
}

static inline void avx_map64_destroy(struct avx_map64 *m) {
    free(m->meta);
    free(m->keys);
}

static inline int avx_map64_insert(struct avx_map64 *m, uint64_t key) {
    if (m->cap == 0) avx64_alloc(m, AVX64_INIT_CAP);
    if (m->count * AVX64_LOAD_DEN >= m->cap * AVX64_LOAD_NUM)
        avx64_grow(m);

    uint64_t h  = avx64_hash(key);
    uint8_t  h2 = avx64_h2(h);
    uint32_t ng = m->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t  *meta = m->meta + gi * 64;
        uint64_t *keys = m->keys + gi * 64;

        __mmask64 mm = avx64_match(meta, h2);
        while (mm) {
            int pos = __builtin_ctzll(mm);
            if (keys[pos] == key) return 0;
            mm &= mm - 1;
        }

        __mmask64 em = avx64_empty(meta);
        if (em) {
            int pos = __builtin_ctzll(em);
            meta[pos] = h2;
            keys[pos] = key;
            m->count++;
            return 1;
        }
        gi = (gi + 1) & (ng - 1);
    }
}

static inline int avx_map64_contains(struct avx_map64 *m, uint64_t key) {
    if (m->cap == 0) return 0;

    uint64_t h  = avx64_hash(key);
    uint8_t  h2 = avx64_h2(h);
    uint32_t ng = m->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t  *meta = m->meta + gi * 64;
        uint64_t *keys = m->keys + gi * 64;

        __mmask64 mm = avx64_match(meta, h2);
        while (mm) {
            int pos = __builtin_ctzll(mm);
            if (keys[pos] == key) return 1;
            mm &= mm - 1;
        }
        if (avx64_empty(meta)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

#endif /* AVX_MAP64_H */
