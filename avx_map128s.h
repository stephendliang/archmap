/*
 * avx_map128s: AVX-512 hash set for 128-bit keys with 16-bit h2 and overflow sentinel
 *
 * Header-only. Interleaved group layout: each group is 576 bytes
 * (64B metadata + 512B keys) in a single contiguous allocation.
 * 31 data slots + 1 overflow sentinel per group.
 * 15-bit h2 gives 1/32768 false positive rate (256x better than 7-bit).
 * Sentinel eliminates SIMD empty check on contains hot path.
 *
 * Group layout (576 bytes):
 *   [0,  64)  uint16_t meta[32]   31 h2 slots + 1 overflow sentinel
 *   [64, 560) avx128s_kv keys[31] 31 × 16B = 496B
 *   [560,568) uint64_t disp       2-bit saturating displacement per slot
 *   [568,576) padding (unused)
 *
 * Displacement bitmap: 2 bits per data slot encode how far the key is
 * displaced from its home group (0/1/2 = exact, 3 = saturated → rehash).
 * Backshift reads displacement instead of rehashing every occupied key,
 * eliminating ~99% of CRC32 calls during deletion.
 *
 * Key width is irrelevant to the SIMD hot path: SIMD operates only on
 * 16-bit h2 metadata, and scalar key comparison fires only on h2 match.
 */
#ifndef AVX_MAP128S_H
#define AVX_MAP128S_H

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define AVX128S_INIT_CAP    32
#define AVX128S_LOAD_NUM    7
#define AVX128S_LOAD_DEN    8
#define AVX128S_DATA_MASK   0x7FFFFFFFu   /* exclude position 31 (sentinel) */
#define AVX128S_GROUP_BYTES 576u          /* 64B meta + 32×16B keys per group */

struct avx128s_kv { uint64_t lo, hi; };

struct avx_map128s {
    char *data;          /* interleaved groups: [meta 64B | keys 512B] × ng */
    uint32_t count;
    uint32_t cap;        /* ng * 32 (includes sentinel positions) */
    uint32_t mask;       /* (cap >> 5) - 1, precomputed for group index */
};

/* --- Hash: chained CRC32 mixer (three 32-bit rounds over both halves) --- */

struct avx128s_h { uint32_t lo, hi; };

static inline struct avx128s_h avx128s_hash(uint64_t klo, uint64_t khi) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, klo);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, khi);
    uint32_t c = (uint32_t)_mm_crc32_u64(b, klo);
    return (struct avx128s_h){b, c};
}

/* --- Metadata encoding --- */

static inline uint16_t avx128s_h2(uint32_t hi) {
    return (uint16_t)((hi >> 17) | 0x8000);
}

static inline uint16_t avx128s_overflow_bit(uint32_t hi) {
    return (uint16_t)(1u << (hi & 15));
}

/* --- Group access helpers --- */

static inline char *avx128s_group(const struct avx_map128s *m, uint32_t gi) {
    return m->data + (size_t)gi * AVX128S_GROUP_BYTES;
}

/* --- Displacement bitmap: 2-bit per slot at byte offset 560 in group --- */

static inline uint64_t *avx128s_disp(char *grp) {
    return (uint64_t *)(grp + 560);
}

static inline uint32_t avx128s_get_disp(uint64_t d, int slot) {
    return (d >> (slot * 2)) & 3;
}

static inline void avx128s_set_disp(uint64_t *dp, int slot, uint32_t val) {
    *dp = (*dp & ~(3ULL << (slot * 2))) | ((uint64_t)val << (slot * 2));
}

/* --- Prefetch helper --- */

static inline void avx_map128s_prefetch(const struct avx_map128s *m,
                                        uint64_t klo, uint64_t khi) {
    uint32_t a  = (uint32_t)_mm_crc32_u64(0, klo);
    uint32_t b  = (uint32_t)_mm_crc32_u64(a, khi);
    uint32_t gi = b & m->mask;
    const char *grp = avx128s_group(m, gi);
    _mm_prefetch(grp, _MM_HINT_T0);        /* metadata cache line */
    _mm_prefetch(grp + 64, _MM_HINT_T0);   /* keys[0..3] */
    _mm_prefetch(grp + 128, _MM_HINT_T0);  /* keys[4..7] */
}

/* --- SIMD helpers --- */

static inline __mmask32 avx128s_match(const uint16_t *meta, uint16_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi16((short)h2);
    return _mm512_cmpeq_epi16_mask(group, needle) & AVX128S_DATA_MASK;
}

static inline __mmask32 avx128s_empty(const uint16_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_testn_epi16_mask(group, group) & AVX128S_DATA_MASK;
}

/* --- Alloc / grow --- */

static size_t avx128s_mapsize(uint32_t cap) {
    size_t raw = (size_t)(cap >> 5) * AVX128S_GROUP_BYTES;
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1); /* round to 2MB */
}

static void avx128s_alloc(struct avx_map128s *m, uint32_t cap) {
    size_t total = avx128s_mapsize(cap);
    /* Try explicit 2MB hugepages with MAP_POPULATE (pre-fault all pages,
     * eliminates minor faults during first-touch and ensures the OS
     * commits physical hugepages immediately rather than on demand). */
    m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                           | MAP_POPULATE, -1, 0);
    if (m->data == MAP_FAILED) {
        /* Fallback: regular pages + THP hint + populate */
        m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS
                               | MAP_POPULATE, -1, 0);
        if (m->data != MAP_FAILED)
            madvise(m->data, total, MADV_HUGEPAGE);
    }
    m->cap   = cap;
    m->mask  = (cap >> 5) - 1;
    m->count = 0;
}

static void avx128s_grow(struct avx_map128s *m) {
    uint32_t old_cap  = m->cap;
    char    *old_data = m->data;
    uint32_t old_ng   = old_cap >> 5;

    avx128s_alloc(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
        const char     *old_grp = old_data + (size_t)g * AVX128S_GROUP_BYTES;
        const uint16_t *om      = (const uint16_t *)old_grp;
        const struct avx128s_kv *ok = (const struct avx128s_kv *)(old_grp + 64);
        for (int s = 0; s < 31; s++) {
            if (!(om[s] & 0x8000)) continue;
            uint64_t klo = ok[s].lo;
            uint64_t khi = ok[s].hi;
            struct avx128s_h h = avx128s_hash(klo, khi);
            uint16_t h2  = avx128s_h2(h.hi);
            uint32_t gi  = h.lo & mask;
            for (;;) {
                char     *grp  = avx128s_group(m, gi);
                uint16_t *base = (uint16_t *)grp;
                struct avx128s_kv *kp = (struct avx128s_kv *)(grp + 64);
                __mmask32 em = avx128s_empty(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = h2;
                    kp[pos]   = (struct avx128s_kv){klo, khi};
                    uint32_t d = (gi - (h.lo & mask)) & mask;
                    avx128s_set_disp(avx128s_disp(grp), pos, d < 3 ? d : 3);
                    m->count++;
                    break;
                }
                base[31] |= avx128s_overflow_bit(h.hi);
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, avx128s_mapsize(old_cap));
}

/* --- Public API --- */

static inline void avx_map128s_init(struct avx_map128s *m) {
    memset(m, 0, sizeof(*m));
}

static inline void avx_map128s_destroy(struct avx_map128s *m) {
    if (m->data) munmap(m->data, avx128s_mapsize(m->cap));
}

static inline int avx_map128s_insert(struct avx_map128s *m,
                                     uint64_t klo, uint64_t khi) {
    if (m->cap == 0) avx128s_alloc(m, AVX128S_INIT_CAP);
    if (m->count * AVX128S_LOAD_DEN >= m->cap * AVX128S_LOAD_NUM)
        avx128s_grow(m);

    struct avx128s_h h = avx128s_hash(klo, khi);
    uint16_t h2 = avx128s_h2(h.hi);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = avx128s_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct avx128s_kv *kp = (struct avx128s_kv *)(grp + 64);

        __mmask32 mm = avx128s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) return 0;
            mm &= mm - 1;
        }

        __mmask32 em = avx128s_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            kp[pos]   = (struct avx128s_kv){klo, khi};
            uint32_t d = (gi - (h.lo & m->mask)) & m->mask;
            avx128s_set_disp(avx128s_disp(grp), pos, d < 3 ? d : 3);
            m->count++;
            return 1;
        }
        base[31] |= avx128s_overflow_bit(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

/* --- Backshift helper: displacement-accelerated probe chain repair --- */

static inline void avx128s_backshift_at(struct avx_map128s *m,
                                         uint32_t gi, int slot) {
    uint32_t mask = m->mask;
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = (gi + 1) & mask;

    for (;;) {
        char *pf_grp = avx128s_group(m, (scan_gi + 2) & mask);
        _mm_prefetch(pf_grp, _MM_HINT_T0);
        _mm_prefetch(pf_grp + 64, _MM_HINT_T0);
        _mm_prefetch(pf_grp + 128, _MM_HINT_T0);

        char *scan_raw = avx128s_group(m, scan_gi);
        uint16_t *scan_base = (uint16_t *)scan_raw;
        struct avx128s_kv *scan_kp = (struct avx128s_kv *)(scan_raw + 64);
        uint64_t *scan_dp = avx128s_disp(scan_raw);
        uint64_t scan_disp = *scan_dp;
        __mmask32 scan_empty = avx128s_empty(scan_base);

        if ((scan_empty & AVX128S_DATA_MASK) == AVX128S_DATA_MASK)
            return; /* fully empty group — chain over */

        /* collect occupied candidates */
        uint32_t cand_homes[31];
        int cand_slots[31];
        int n_cand = 0;

        __mmask32 occupied = (~scan_empty) & AVX128S_DATA_MASK;
        while (occupied) {
            cand_slots[n_cand++] = __builtin_ctz(occupied);
            occupied &= occupied - 1;
        }

        /* resolve home groups from displacement — rehash only saturated */
        for (int j = 0; j < n_cand; j++) {
            uint32_t d = avx128s_get_disp(scan_disp, cand_slots[j]);
            if (d < 3)
                cand_homes[j] = (scan_gi - d) & mask;
            else
                cand_homes[j] = avx128s_hash(scan_kp[cand_slots[j]].lo,
                                              scan_kp[cand_slots[j]].hi).lo
                                & mask;
        }

        /* find first movable candidate */
        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            if (((hole_gi - cand_homes[j]) & mask) <
                ((scan_gi - cand_homes[j]) & mask)) {
                char *hole_raw = avx128s_group(m, hole_gi);
                uint16_t *hole_base = (uint16_t *)hole_raw;
                struct avx128s_kv *hole_kp =
                    (struct avx128s_kv *)(hole_raw + 64);
                uint64_t *hole_dp = avx128s_disp(hole_raw);

                hole_base[hole_slot] = scan_base[cand_slots[j]];
                hole_kp[hole_slot] = scan_kp[cand_slots[j]];

                /* update displacement for new position */
                uint32_t new_d = (hole_gi - cand_homes[j]) & mask;
                avx128s_set_disp(hole_dp, hole_slot, new_d < 3 ? new_d : 3);

                /* clear source slot */
                scan_base[cand_slots[j]] = 0;
                scan_kp[cand_slots[j]] = (struct avx128s_kv){0, 0};
                avx128s_set_disp(scan_dp, cand_slots[j], 0);

                if (scan_empty) return;

                hole_gi = scan_gi;
                hole_slot = cand_slots[j];
                moved = 1;
                break;
            }
        }

        if (!moved && scan_empty) return;
        scan_gi = (scan_gi + 1) & mask;
    }
}

/* --- Backshift delete: find key via overflow chain, delete + backshift --- */

static inline int avx_map128s_delete(struct avx_map128s *m,
                                     uint64_t klo, uint64_t khi) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct avx128s_h h = avx128s_hash(klo, khi);
    uint16_t h2 = avx128s_h2(h.hi);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = avx128s_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct avx128s_kv *kp = (struct avx128s_kv *)(grp + 64);

        __mmask32 mm = avx128s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) {
                /* Found. Check empties BEFORE zeroing. */
                __mmask32 em = avx128s_empty(base);
                base[pos] = 0;
                kp[pos] = (struct avx128s_kv){0, 0};
                avx128s_set_disp(avx128s_disp(grp), pos, 0);
                m->count--;
                if (!em) avx128s_backshift_at(m, gi, pos);
                return 1;
            }
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int avx_map128s_contains(struct avx_map128s *m,
                                       uint64_t klo, uint64_t khi) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct avx128s_h h = avx128s_hash(klo, khi);
    uint16_t h2 = avx128s_h2(h.hi);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = avx128s_group(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const struct avx128s_kv *kp = (const struct avx128s_kv *)(grp + 64);

        __mmask32 mm = avx128s_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) return 1;
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

#endif /* AVX_MAP128S_H */
