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

/* prefetch home group + overflow group for multi-probe chains */
static inline void avx_map64_prefetch2(struct avx_map64 *m, uint64_t key) {
    uint32_t gi = avx64_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
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

/* --- Backshift helper: repair probe chain after deletion --- */

static inline void avx64_backshift_at(struct avx_map64 *m,
                                       uint32_t gi, int slot) {
    /* Pull displaced keys toward their home group.
     *
     * Invariant: for key K at group G with home H, groups H..G-1
     * are all completely full.  We just punched a hole in a full
     * group, so any key downstream whose probe path crossed this
     * group is now unreachable via contains (which stops at the
     * first group with an empty slot).  Walk forward and pull
     * one key per group back to fill each successive hole.
     */
    uint32_t mask = m->mask;
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = (gi + 1) & mask;

    for (;;) {
        /* prefetch 2 groups ahead — sequential scan is predictable */
        uint32_t pf_gi = (scan_gi + 2) & mask;
        _mm_prefetch((const char *)(m->keys + (pf_gi << 3)), _MM_HINT_T0);

        uint64_t *scan_grp = m->keys + (scan_gi << 3);
        __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
        __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

        if (scan_empty == 0xFF) return; /* fully empty group — chain over */

        /* pre-hash all occupied keys — independent CRC32s
         * pipeline at 1-cycle throughput via OoO execution */
        uint64_t cand_keys[8];
        uint32_t cand_homes[8];
        int cand_slots[8];
        int n_cand = 0;

        for (__mmask8 todo = (~scan_empty) & 0xFF; todo; todo &= todo - 1) {
            int s = __builtin_ctz(todo);
            cand_keys[n_cand] = scan_grp[s];
            cand_slots[n_cand] = s;
            n_cand++;
        }

        for (int j = 0; j < n_cand; j++)
            cand_homes[j] = avx64_hash(cand_keys[j]) & mask;

        /* find first movable candidate */
        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            if (((hole_gi - cand_homes[j]) & mask) <
                ((scan_gi - cand_homes[j]) & mask)) {
                uint64_t *hole_grp = m->keys + (hole_gi << 3);
                hole_grp[hole_slot] = cand_keys[j];
                scan_grp[cand_slots[j]] = 0;

                /* scan group already had empties → chain ends here */
                if (scan_empty) return;

                hole_gi = scan_gi;
                hole_slot = cand_slots[j];
                moved = 1;
                break;
            }
        }

        if (!moved && scan_empty) return; /* empties, no candidates — done */
        scan_gi = (scan_gi + 1) & mask;
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
            int slot = __builtin_ctz(mm);
            grp[slot] = 0;
            m->count--;
            /* group was non-full before delete → no probe chain to fix */
            if (!empty) avx64_backshift_at(m, gi, slot);
            return 1;
        }
        if (empty) return 0; /* key not found */
        gi = (gi + 1) & mask;
    }
}

/* --- Unified op: single probe loop, branchless dispatch ---
 *
 * Eliminates the 3-way switch dispatch branch that causes ~11%
 * misprediction in mixed workloads.  One probe loop handles all
 * three operations; op-dependent logic runs only at terminal
 * points (found / not-found), where branches are biased and
 * well-predicted.  The compiler generates a branchless cmp+adc
 * for the common contains path.
 *
 * op: 0=contains, 1=insert, 2=delete
 */

static inline int avx_map64_op(struct avx_map64 *m, uint64_t key, int op) {
    /* growth check — only for insert (rare), skipped for contains/delete */
    if (__builtin_expect(m->cap == 0, 0)) {
        if (op == 1) avx64_alloc(m, AVX64_INIT_CAP);
        else return 0;
    }
    if (__builtin_expect(op == 1 &&
            m->count * AVX64_LOAD_DEN >= m->cap * AVX64_LOAD_NUM, 0))
        avx64_grow(m);

    uint32_t gi   = avx64_hash(key) & m->mask;
    uint32_t mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        __mmask8 mm    = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);

        if (mm) {
            /* key found: delete path rare, contains/insert branchless */
            if (__builtin_expect(op == 2, 0)) {
                int slot = __builtin_ctz(mm);
                grp[slot] = 0;
                m->count--;
                if (!empty) avx64_backshift_at(m, gi, slot);
                return 1;
            }
            return op == 0; /* 1 for contains, 0 for insert-dup — no branch */
        }
        if (empty) {
            /* key not found: insert path rare, contains/delete branchless */
            if (__builtin_expect(op == 1, 0)) {
                grp[__builtin_ctz(empty)] = key;
                m->count++;
                return 1;
            }
            return 0; /* 0 for contains-miss and delete-miss — no branch */
        }
        gi = (gi + 1) & mask;
    }
}

#endif /* AVX_MAP64_H */
