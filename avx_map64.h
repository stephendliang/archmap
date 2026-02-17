/* avx_map64: Zero-metadata direct-key AVX-512 hash set for uint64_t

Header-only. Keys stored directly in 8-wide groups (one cache line).
vpcmpeqq compares all 8 keys at once — zero false positives, no metadata,
no scalar verification. Key=0 reserved as empty sentinel. */
#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define AVX64_INIT_CAP  64  // 8 groups
#define AVX64_LOAD_NUM  3
#define AVX64_LOAD_DEN  4   // 75% load factor

struct avx_map64 {
    uint64_t *keys;     // aligned, zero = empty
    uint32_t count;
    uint32_t cap;       // ng * 8
    uint32_t mask;      // (cap >> 3) - 1
};

// hash: CRC32 integer mixer
#define avx64_hash(key) ((uint32_t)_mm_crc32_u64(0, (key)))

// SIMD match/empty masks
#define avx64_match(grp, key) \
    _mm512_cmpeq_epi64_mask(_mm512_load_si512((const __m512i *)(grp)), \
                            _mm512_set1_epi64((long long)(key)))

#define avx64_empty(grp) \
    _mm512_cmpeq_epi64_mask(_mm512_load_si512((const __m512i *)(grp)), \
                            _mm512_setzero_si512())

// prefetch home group
#define avx_map64_prefetch(m, key) do { \
    uint32_t gi_ = avx64_hash(key) & (m)->mask; \
    _mm_prefetch((const char *)((m)->keys + (gi_ << 3)), _MM_HINT_T0); \
} while (0)

// prefetch home + overflow group
static inline void avx_map64_prefetch2(struct avx_map64 *m, uint64_t key) {
    uint32_t gi = avx64_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

// round byte size up to 2MB
#define avx64_mapsize(cap) \
    (((size_t)(cap) * sizeof(uint64_t) + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1))

#define avx_map64_init(m)    memset((m), 0, sizeof(*(m)))
#define avx_map64_destroy(m) do { \
    if ((m)->keys) munmap((m)->keys, avx64_mapsize((m)->cap)); \
} while (0)

static void avx64_alloc(struct avx_map64 *m, uint32_t cap) {
    size_t total = avx64_mapsize(cap);
    // try explicit 2MB hugepages with MAP_POPULATE (pre-fault)
    m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (m->keys == MAP_FAILED) {
        // fallback: regular pages + THP hint
        m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (m->keys != MAP_FAILED)
            madvise(m->keys, total, MADV_HUGEPAGE);
    }
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
}

static void avx64_grow(struct avx_map64 *m) {
    uint32_t old_cap = m->cap;
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

// public API

static inline int avx_map64_insert(struct avx_map64 *m, uint64_t key) {
    if (m->cap == 0) avx64_alloc(m, AVX64_INIT_CAP);
    if (m->count * AVX64_LOAD_DEN >= m->cap * AVX64_LOAD_NUM) avx64_grow(m);
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
        if (_mm512_cmpeq_epi64_mask(group, needle)) return 1;
        if (_mm512_testn_epi64_mask(group, group))  return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* backshift: repair probe chain after deletion.
pulls displaced keys toward their home group so contains()
(which stops at the first empty slot) remains correct. */
static inline void avx64_backshift_at(struct avx_map64 *m, uint32_t gi, int slot) {
    uint32_t mask = m->mask;
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = (gi + 1) & mask;

    for (;;) {
        uint32_t pf_gi = (scan_gi + 2) & mask;
        _mm_prefetch((const char *)(m->keys + (pf_gi << 3)), _MM_HINT_T0);

        uint64_t *scan_grp = m->keys + (scan_gi << 3);
        __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
        __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

        if (scan_empty == 0xFF) return; // fully empty — chain over

        // hash all occupied keys, find movable candidate
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

        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            if (((hole_gi - cand_homes[j]) & mask) < ((scan_gi - cand_homes[j]) & mask)) {
                uint64_t *hole_grp = m->keys + (hole_gi << 3);
                hole_grp[hole_slot] = cand_keys[j];
                scan_grp[cand_slots[j]] = 0;
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

static inline int avx_map64_delete(struct avx_map64 *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = avx64_hash(key) & m->mask, mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);
        if (mm) {
            int slot = __builtin_ctz(mm);
            grp[slot] = 0;
            m->count--;
            if (!empty) avx64_backshift_at(m, gi, slot);
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & mask;
    }
}

// unified op: single probe loop. op: 0=contains, 1=insert, 2=delete
static inline int avx_map64_op(struct avx_map64 *m, uint64_t key, int op) {
    if (__builtin_expect(m->cap == 0, 0)) {
        if (op == 1) avx64_alloc(m, AVX64_INIT_CAP);
        else return 0;
    }
    if (__builtin_expect(op == 1 && m->count * AVX64_LOAD_DEN >= m->cap * AVX64_LOAD_NUM, 0)) avx64_grow(m);
    uint32_t gi = avx64_hash(key) & m->mask, mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        __mmask8 mm = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);
        if (mm) {
            if (__builtin_expect(op == 2, 0)) {
                int slot = __builtin_ctz(mm);
                grp[slot] = 0;
                m->count--;
                if (!empty) avx64_backshift_at(m, gi, slot);
                return 1;
            }
            return op == 0; // 1 for contains, 0 for insert-dup
        }
        if (empty) {
            if (__builtin_expect(op == 1, 0)) {
                grp[__builtin_ctz(empty)] = key;
                m->count++;
                return 1;
            }
            return 0;
        }
        gi = (gi + 1) & mask;
    }
}
