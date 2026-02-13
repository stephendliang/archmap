/*
 * avx_map64s: AVX-512 hash set with 16-bit h2 and overflow sentinel
 *
 * Header-only. Interleaved group layout: each group is 320 bytes
 * (64B metadata + 256B keys) in a single contiguous allocation.
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

#define AVX64S_INIT_CAP    32
#define AVX64S_LOAD_NUM    7
#define AVX64S_LOAD_DEN    8
#define AVX64S_DATA_MASK   0x7FFFFFFFu   /* exclude position 31 (sentinel) */
#define AVX64S_GROUP_BYTES 320u          /* 64B meta + 256B keys per group */

struct avx_map64s {
    char *data;          /* interleaved groups: [meta 64B | keys 256B] × ng */
    uint32_t count;
    uint32_t cap;        /* ng * 32 (includes sentinel positions) */
};

/* --- Hash: chained CRC32 mixer (full 64-bit output) --- */

static inline uint64_t avx64s_hash(uint64_t key) {
    uint64_t lo = _mm_crc32_u64(0, key);
    uint64_t hi = _mm_crc32_u64(lo, key);
    return lo | (hi << 32);
}

/* --- Metadata encoding --- */

static inline uint16_t avx64s_h2(uint64_t hash) {
    return (uint16_t)((hash >> 49) | 0x8000);
}

static inline uint16_t avx64s_overflow_bit(uint64_t hash) {
    return (uint16_t)(1u << ((hash >> 32) & 15));
}

/* --- Group access helpers --- */

static inline char *avx64s_group(const struct avx_map64s *m, uint32_t gi) {
    return m->data + (size_t)gi * AVX64S_GROUP_BYTES;
}

/* --- Prefetch helper --- */

static inline void avx_map64s_prefetch(const struct avx_map64s *m, uint64_t key) {
    uint32_t lo = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t gi = lo & ((m->cap >> 5) - 1);
    const char *grp = avx64s_group(m, gi);
    _mm_prefetch(grp, _MM_HINT_T0);        /* metadata cache line */
    _mm_prefetch(grp + 64, _MM_HINT_T0);   /* keys[0..7] */
    _mm_prefetch(grp + 128, _MM_HINT_T0);  /* keys[8..15] */
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
    uint32_t ng = cap >> 5;
    size_t total = (size_t)ng * AVX64S_GROUP_BYTES;
    m->data = (char *)aligned_alloc(64, total);
    memset(m->data, 0, total);
    madvise(m->data, total, MADV_HUGEPAGE);
    m->cap   = cap;
    m->count = 0;
}

static void avx64s_grow(struct avx_map64s *m) {
    uint32_t old_cap  = m->cap;
    char    *old_data = m->data;
    uint32_t old_ng   = old_cap >> 5;

    avx64s_alloc(m, old_cap * 2);
    uint32_t ng = m->cap >> 5;

    for (uint32_t g = 0; g < old_ng; g++) {
        const char     *old_grp = old_data + (size_t)g * AVX64S_GROUP_BYTES;
        const uint16_t *om      = (const uint16_t *)old_grp;
        const uint64_t *ok      = (const uint64_t *)(old_grp + 64);
        for (int s = 0; s < 31; s++) {
            if (!(om[s] & 0x8000)) continue;
            uint64_t key = ok[s];
            uint64_t h   = avx64s_hash(key);
            uint16_t h2  = avx64s_h2(h);
            uint32_t gi  = (uint32_t)h & (ng - 1);
            for (;;) {
                char     *grp  = avx64s_group(m, gi);
                uint16_t *base = (uint16_t *)grp;
                uint64_t *kp   = (uint64_t *)(grp + 64);
                __mmask32 em = avx64s_empty(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = h2;
                    kp[pos]   = key;
                    m->count++;
                    break;
                }
                base[31] |= avx64s_overflow_bit(h);
                gi = (gi + 1) & (ng - 1);
            }
        }
    }
    free(old_data);
}

/* --- Public API --- */

static inline void avx_map64s_init(struct avx_map64s *m) {
    memset(m, 0, sizeof(*m));
}

static inline void avx_map64s_destroy(struct avx_map64s *m) {
    free(m->data);
}

static inline int avx_map64s_insert(struct avx_map64s *m, uint64_t key) {
    if (m->cap == 0) avx64s_alloc(m, AVX64S_INIT_CAP);
    if (m->count * AVX64S_LOAD_DEN >= m->cap * AVX64S_LOAD_NUM)
        avx64s_grow(m);

    uint64_t h  = avx64s_hash(key);
    uint16_t h2 = avx64s_h2(h);
    uint32_t ng = m->cap >> 5;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        char     *grp  = avx64s_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        uint64_t *kp   = (uint64_t *)(grp + 64);

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
        base[31] |= avx64s_overflow_bit(h);
        gi = (gi + 1) & (ng - 1);
    }
}

static inline int avx_map64s_contains(struct avx_map64s *m, uint64_t key) {
    if (m->cap == 0) return 0;

    uint64_t h  = avx64s_hash(key);
    uint16_t h2 = avx64s_h2(h);
    uint32_t ng = m->cap >> 5;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        const char     *grp  = avx64s_group(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const uint64_t *kp   = (const uint64_t *)(grp + 64);

        __mmask32 mm = avx64s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos] == key) return 1;
            mm &= mm - 1;
        }
        if (!(base[31] & avx64s_overflow_bit(h))) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

#endif /* AVX_MAP64S_H */
