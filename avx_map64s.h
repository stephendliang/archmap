/*
 * avx_map64s: AVX-512 hash set with 16-bit h2 and overflow sentinel
 *
 * Header-only. 32 x uint16_t metadata per group (one cache line).
 * 31 data slots + 1 overflow sentinel per group.
 * 15-bit h2 gives 1/32768 false positive rate (256x better than 7-bit).
 * Sentinel eliminates SIMD empty check on contains hot path.
 */
#ifndef AVX_MAP64S_H
#define AVX_MAP64S_H

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define AVX64S_INIT_CAP   32
#define AVX64S_LOAD_NUM   7
#define AVX64S_LOAD_DEN   8
#define AVX64S_DATA_MASK  0x7FFFFFFFu   /* exclude position 31 (sentinel) */

struct avx_map64s {
    uint16_t *meta;
    uint64_t *keys;
    uint32_t count;
    uint32_t cap;       /* ng * 32 (includes sentinel positions) */
};

/* --- Hash: same fast integer mixer as avx_map64 --- */

static inline uint64_t avx64s_hash(uint64_t key) {
    key ^= key >> 23;
    key *= 0x2127599bf4325c37ULL;
    key ^= key >> 47;
    return key;
}

/* --- Metadata encoding --- */

static inline uint16_t avx64s_h2(uint64_t hash) {
    return (uint16_t)((hash >> 49) | 0x8000);
}

static inline uint16_t avx64s_overflow_bit(uint64_t hash) {
    return (uint16_t)(1u << ((hash >> 32) & 15));
}

/* --- Prefetch helper --- */

static inline void avx_map64s_prefetch(struct avx_map64s *m, uint64_t key) {
    uint64_t h = avx64s_hash(key);
    uint32_t gi = (uint32_t)h & ((m->cap >> 5) - 1);
    _mm_prefetch((const char *)(m->meta + (gi << 5)), _MM_HINT_T0);
}

/* --- SIMD helpers --- */

static inline __mmask32 avx64s_match(const uint16_t *meta, uint16_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi16((short)h2);
    return _mm512_mask_cmpeq_epi16_mask(AVX64S_DATA_MASK, group, needle);
}

static inline __mmask32 avx64s_empty(const uint16_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_mask_cmpeq_epi16_mask(AVX64S_DATA_MASK, group, _mm512_setzero_si512());
}

/* --- Alloc / grow --- */

static void avx64s_alloc(struct avx_map64s *m, uint32_t cap) {
    m->meta = (uint16_t *)aligned_alloc(64, cap * sizeof(uint16_t));
    memset(m->meta, 0, cap * sizeof(uint16_t));
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    madvise(m->meta, cap * sizeof(uint16_t), MADV_HUGEPAGE);
    madvise(m->keys, cap * sizeof(uint64_t), MADV_HUGEPAGE);
    m->cap   = cap;
    m->count = 0;
}

static void avx64s_grow(struct avx_map64s *m) {
    uint32_t  old_cap  = m->cap;
    uint16_t *old_meta = m->meta;
    uint64_t *old_keys = m->keys;

    avx64s_alloc(m, old_cap * 2);
    uint32_t ng = m->cap >> 5;

    for (uint32_t i = 0; i < old_cap; i++) {
        if ((i & 31) == 31) continue;           /* skip sentinel slots */
        if (!(old_meta[i] & 0x8000)) continue;  /* skip empty */
        uint64_t key = old_keys[i];
        uint64_t h   = avx64s_hash(key);
        uint16_t h2  = avx64s_h2(h);
        uint16_t ob  = avx64s_overflow_bit(h);
        uint32_t gi  = (uint32_t)h & (ng - 1);
        for (;;) {
            uint16_t *base = m->meta + (gi << 5);
            uint64_t *kp   = m->keys + (gi << 5);
            __mmask32 em = avx64s_empty(base);
            if (em) {
                int pos = __builtin_ctz(em);
                base[pos] = h2;
                kp[pos]   = key;
                m->count++;
                break;
            }
            base[31] |= ob;
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

/* --- Public API --- */

static inline void avx_map64s_init(struct avx_map64s *m) {
    memset(m, 0, sizeof(*m));
}

static inline void avx_map64s_destroy(struct avx_map64s *m) {
    free(m->meta);
    free(m->keys);
}

static inline int avx_map64s_insert(struct avx_map64s *m, uint64_t key) {
    if (m->cap == 0) avx64s_alloc(m, AVX64S_INIT_CAP);
    if (m->count * AVX64S_LOAD_DEN >= m->cap * AVX64S_LOAD_NUM)
        avx64s_grow(m);

    uint64_t h  = avx64s_hash(key);
    uint16_t h2 = avx64s_h2(h);
    uint16_t ob = avx64s_overflow_bit(h);
    uint32_t ng = m->cap >> 5;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint16_t *base = m->meta + (gi << 5);
        uint64_t *kp   = m->keys + (gi << 5);

        __mmask32 mm = avx64s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos] == key) return 0;
            mm &= mm - 1;
        }

        __mmask32 em = avx64s_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            kp[pos]   = key;
            m->count++;
            return 1;
        }
        base[31] |= ob;
        gi = (gi + 1) & (ng - 1);
    }
}

static inline int avx_map64s_contains(struct avx_map64s *m, uint64_t key) {
    if (m->cap == 0) return 0;

    uint64_t h  = avx64s_hash(key);
    uint16_t h2 = avx64s_h2(h);
    uint16_t ob = avx64s_overflow_bit(h);
    uint32_t ng = m->cap >> 5;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint16_t *base = m->meta + (gi << 5);
        uint64_t *kp   = m->keys + (gi << 5);

        __mmask32 mm = avx64s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos] == key) return 1;
            mm &= mm - 1;
        }
        if (!(base[31] & ob)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

#endif /* AVX_MAP64S_H */
