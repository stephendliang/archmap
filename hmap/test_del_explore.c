/*
 * Deletion strategy exploration for avx_map64.
 *
 * Variants tested:
 *   A) baseline   — current pipelined-CRC32 backshift
 *   B) tombstone  — key=1 sentinel, no backshift, periodic compaction
 *   C) hybrid     — 1-group bounded backshift, tombstone fallback
 *   D) disp-bitmap— 1 byte/group displacement map, skip all-home groups
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *       -o test_del_explore test_del_explore.c -lm
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>

/* ================================================================
 * Shared definitions
 * ================================================================ */

#define INIT_CAP   64
#define LOAD_NUM   3
#define LOAD_DEN   4
#define PF_DIST    12
#define TOMBSTONE  1ULL

static inline uint32_t map_hash(uint64_t key) {
    return (uint32_t)_mm_crc32_u64(0, key);
}

static inline size_t mapsize(uint32_t cap) {
    size_t raw = (size_t)cap * sizeof(uint64_t);
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static uint64_t *map_mmap(uint32_t cap) {
    size_t total = mapsize(cap);
    uint64_t *p = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p == MAP_FAILED)
        p = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p;
}

/* ================================================================
 * xoshiro256** PRNG
 * ================================================================ */

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}

static void seed_rng(void) {
    uint64_t s = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * VARIANT A: Baseline (avx_map64 verbatim from header)
 * ================================================================ */

#include "simd_map64.h"

/* prefetch wrappers already in header */

/* ================================================================
 * Instrumented baseline — same algorithm, gathers statistics
 * ================================================================ */

struct del_stats {
    uint64_t total_deletes;
    uint64_t no_backshift;       /* group had empties → early return */
    uint64_t backshift_triggered;
    uint64_t total_groups_scanned;
    uint64_t total_keys_hashed;
    uint64_t total_keys_displaced; /* hashed key was NOT at its home group */
    uint64_t total_keys_moved;
    uint64_t chain_hist[17];     /* chain length 0..15, 16=overflow */
};

static struct del_stats g_stats;

static int instrumented_delete(struct simd_map64 *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    uint32_t gi   = map_hash(key) & m->mask;
    uint32_t mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm    = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);

        if (mm) {
            grp[__builtin_ctz(mm)] = 0;
            m->count--;
            g_stats.total_deletes++;

            if (empty) {
                g_stats.no_backshift++;
                g_stats.chain_hist[0]++;
                return 1;
            }

            g_stats.backshift_triggered++;
            uint32_t hole_gi = gi;
            int hole_slot = __builtin_ctz(mm);
            uint32_t scan_gi = (gi + 1) & mask;
            int chain_len = 0;

            for (;;) {
                uint64_t *scan_grp = m->keys + (scan_gi << 3);
                __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
                __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

                if (scan_empty == 0xFF) {
                    g_stats.chain_hist[chain_len < 16 ? chain_len : 16]++;
                    break;
                }

                chain_len++;
                g_stats.total_groups_scanned++;

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

                g_stats.total_keys_hashed += n_cand;

                for (int j = 0; j < n_cand; j++) {
                    cand_homes[j] = map_hash(cand_keys[j]) & mask;
                    if (cand_homes[j] != scan_gi)
                        g_stats.total_keys_displaced++;
                }

                int moved = 0;
                for (int j = 0; j < n_cand; j++) {
                    if (((hole_gi - cand_homes[j]) & mask) < ((scan_gi - cand_homes[j]) & mask)) {
                        uint64_t *hole_grp = m->keys + (hole_gi << 3);
                        hole_grp[hole_slot] = cand_keys[j];
                        scan_grp[cand_slots[j]] = 0;
                        g_stats.total_keys_moved++;

                        if (scan_empty) {
                            g_stats.chain_hist[chain_len < 16 ? chain_len : 16]++;
                            return 1;
                        }
                        hole_gi = scan_gi;
                        hole_slot = cand_slots[j];
                        moved = 1;
                        break;
                    }
                }

                if (!moved && scan_empty) {
                    g_stats.chain_hist[chain_len < 16 ? chain_len : 16]++;
                    break;
                }
                scan_gi = (scan_gi + 1) & mask;
            }
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & mask;
    }
}

/* ================================================================
 * VARIANT B: Pure tombstone (no backshift, periodic compaction)
 * ================================================================ */

struct avx_tomb {
    uint64_t *keys;
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
    uint32_t n_tombstones;
};

static void tomb_alloc(struct avx_tomb *m, uint32_t cap) {
    m->keys = map_mmap(cap);
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
    m->n_tombstones = 0;
}

static inline void tomb_init(struct avx_tomb *m) {
    memset(m, 0, sizeof(*m));
}

static inline void tomb_destroy(struct avx_tomb *m) {
    if (m->keys) munmap(m->keys, mapsize(m->cap));
}

static inline void tomb_prefetch(struct avx_tomb *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static inline void tomb_prefetch2(struct avx_tomb *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static void tomb_rehash(struct avx_tomb *m, uint32_t new_cap);

static inline int tomb_contains(struct avx_tomb *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle))   return 1;
        if (_mm512_testn_epi64_mask(group, group))     return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int tomb_insert(struct avx_tomb *m, uint64_t key) {
    if (m->cap == 0) tomb_alloc(m, INIT_CAP);
    if (m->count * LOAD_DEN >= m->cap * LOAD_NUM)
        tomb_rehash(m, m->cap * 2);
    else if ((m->count + m->n_tombstones) * 8 >= m->cap * 7)
        tomb_rehash(m, m->cap);  /* compact at 87.5% effective load */

    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    __m512i tomb_v = _mm512_set1_epi64((long long)TOMBSTONE);
    int ins_gi = -1, ins_slot = -1;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);

        if (_mm512_cmpeq_epi64_mask(group, needle)) return 0;

        /* track first available slot (tombstone or empty) */
        if (ins_gi < 0) {
            __mmask8 avail = _mm512_testn_epi64_mask(group, group)
                           | _mm512_cmpeq_epi64_mask(group, tomb_v);
            if (avail) {
                ins_gi = (int)gi;
                ins_slot = __builtin_ctz(avail);
            }
        }

        /* true empty → key not present */
        if (_mm512_testn_epi64_mask(group, group)) break;

        gi = (gi + 1) & m->mask;
    }

    uint64_t *grp = m->keys + (ins_gi << 3);
    if (grp[ins_slot] == TOMBSTONE) m->n_tombstones--;
    grp[ins_slot] = key;
    m->count++;
    return 1;
}

static inline int tomb_delete(struct avx_tomb *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm = _mm512_cmpeq_epi64_mask(group, needle);
        if (mm) {
            grp[__builtin_ctz(mm)] = TOMBSTONE;
            m->count--;
            m->n_tombstones++;
            return 1;
        }
        if (_mm512_testn_epi64_mask(group, group)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static void tomb_rehash(struct avx_tomb *m, uint32_t new_cap) {
    uint32_t old_cap = m->cap;
    uint64_t *old_keys = m->keys;

    tomb_alloc(m, new_cap);
    uint32_t mask = m->mask;

    for (uint32_t i = 0; i < old_cap; i++) {
        uint64_t key = old_keys[i];
        if (!key || key == TOMBSTONE) continue;
        uint32_t gi = map_hash(key) & mask;
        for (;;) {
            uint64_t *grp = m->keys + (gi << 3);
            __mmask8 em = _mm512_testn_epi64_mask(
                _mm512_load_si512((const __m512i *)grp),
                _mm512_load_si512((const __m512i *)grp));
            if (em) {
                grp[__builtin_ctz(em)] = key;
                m->count++;
                break;
            }
            gi = (gi + 1) & mask;
        }
    }
    munmap(old_keys, mapsize(old_cap));
}

/* ================================================================
 * VARIANT C: Hybrid — 1-group bounded backshift + tombstone fallback
 *
 * Strategy: if the NEXT group after deletion has empties, do a quick
 * 1-group backshift (cheap, bounded cost). Otherwise leave a tombstone
 * and skip multi-group chain walking.
 * ================================================================ */

struct avx_hybrid {
    uint64_t *keys;
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
    uint32_t n_tombstones;
};

static void hybrid_alloc(struct avx_hybrid *m, uint32_t cap) {
    m->keys = map_mmap(cap);
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
    m->n_tombstones = 0;
}

static inline void hybrid_init(struct avx_hybrid *m) {
    memset(m, 0, sizeof(*m));
}

static inline void hybrid_destroy(struct avx_hybrid *m) {
    if (m->keys) munmap(m->keys, mapsize(m->cap));
}

static inline void hybrid_prefetch(struct avx_hybrid *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static inline void hybrid_prefetch2(struct avx_hybrid *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static void hybrid_rehash(struct avx_hybrid *m, uint32_t new_cap);

/* contains: identical to tombstone (stops at true empty, skips tombstones) */
static inline int hybrid_contains(struct avx_hybrid *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle))   return 1;
        if (_mm512_testn_epi64_mask(group, group))     return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* insert: identical to tombstone (reuse tombstone slots) */
static inline int hybrid_insert(struct avx_hybrid *m, uint64_t key) {
    if (m->cap == 0) hybrid_alloc(m, INIT_CAP);
    if (m->count * LOAD_DEN >= m->cap * LOAD_NUM)
        hybrid_rehash(m, m->cap * 2);
    else if ((m->count + m->n_tombstones) * 8 >= m->cap * 7)
        hybrid_rehash(m, m->cap);

    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    __m512i tomb_v = _mm512_set1_epi64((long long)TOMBSTONE);
    int ins_gi = -1, ins_slot = -1;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle)) return 0;
        if (ins_gi < 0) {
            __mmask8 avail = _mm512_testn_epi64_mask(group, group)
                           | _mm512_cmpeq_epi64_mask(group, tomb_v);
            if (avail) {
                ins_gi = (int)gi;
                ins_slot = __builtin_ctz(avail);
            }
        }
        if (_mm512_testn_epi64_mask(group, group)) break;
        gi = (gi + 1) & m->mask;
    }

    uint64_t *grp = m->keys + (ins_gi << 3);
    if (grp[ins_slot] == TOMBSTONE) m->n_tombstones--;
    grp[ins_slot] = key;
    m->count++;
    return 1;
}

/* delete: 1-group bounded backshift, then tombstone fallback */
static inline int hybrid_delete(struct avx_hybrid *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    uint32_t gi   = map_hash(key) & m->mask;
    uint32_t mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm    = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);

        if (mm) {
            int slot = __builtin_ctz(mm);
            grp[slot] = 0;
            m->count--;

            /* group had empties → no chain to fix */
            if (empty) return 1;

            /* peek at next group */
            uint32_t scan_gi = (gi + 1) & mask;
            uint64_t *scan_grp = m->keys + (scan_gi << 3);
            __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
            __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

            if (scan_empty == 0xFF) return 1; /* next group empty → done */

            /* try 1-group backshift: hash occupied keys, find candidate */
            /* also skip tombstones in the scan group */
            __m512i tomb_v = _mm512_set1_epi64((long long)TOMBSTONE);
            __mmask8 tomb_mask = _mm512_cmpeq_epi64_mask(scan_group, tomb_v);
            __mmask8 real_occ = (~scan_empty & ~tomb_mask) & 0xFF;

            if (real_occ) {
                uint64_t cand_keys[8];
                uint32_t cand_homes[8];
                int cand_slots[8];
                int n_cand = 0;

                for (__mmask8 todo = real_occ; todo; todo &= todo - 1) {
                    int s = __builtin_ctz(todo);
                    cand_keys[n_cand] = scan_grp[s];
                    cand_slots[n_cand] = s;
                    n_cand++;
                }
                for (int j = 0; j < n_cand; j++)
                    cand_homes[j] = map_hash(cand_keys[j]) & mask;

                for (int j = 0; j < n_cand; j++) {
                    if (((gi - cand_homes[j]) & mask) < ((scan_gi - cand_homes[j]) & mask)) {
                        grp[slot] = cand_keys[j];
                        /* leave a real empty (or tombstone) at scan */
                        if (scan_empty) {
                            /* scan group has real empties → no key probed
                             * past it, safe to leave real empty */
                            scan_grp[cand_slots[j]] = 0;
                        } else {
                            /* scan group was fully occupied → leave tombstone
                             * to preserve chain for keys further downstream */
                            scan_grp[cand_slots[j]] = TOMBSTONE;
                            m->n_tombstones++;
                        }
                        return 1;
                    }
                }
            }

            /* no candidate found in 1 group — tombstone the hole */
            grp[slot] = TOMBSTONE;
            m->n_tombstones++;
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & mask;
    }
}

static void hybrid_rehash(struct avx_hybrid *m, uint32_t new_cap) {
    uint32_t old_cap = m->cap;
    uint64_t *old_keys = m->keys;
    hybrid_alloc(m, new_cap);
    uint32_t mask = m->mask;
    for (uint32_t i = 0; i < old_cap; i++) {
        uint64_t key = old_keys[i];
        if (!key || key == TOMBSTONE) continue;
        uint32_t gi = map_hash(key) & mask;
        for (;;) {
            uint64_t *grp = m->keys + (gi << 3);
            __mmask8 em = _mm512_testn_epi64_mask(
                _mm512_load_si512((const __m512i *)grp),
                _mm512_load_si512((const __m512i *)grp));
            if (em) { grp[__builtin_ctz(em)] = key; m->count++; break; }
            gi = (gi + 1) & mask;
        }
    }
    munmap(old_keys, mapsize(old_cap));
}

/* ================================================================
 * VARIANT D: Displacement bitmap
 *
 * 1 byte per group: bit i = 1 means slot i holds a key that is NOT
 * at its home group. During backshift, skip groups where disp==0
 * (all keys at home → none can move backward).
 * ================================================================ */

struct avx_disp {
    uint64_t *keys;
    uint8_t  *disp;     /* 1 byte per group */
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
};

static void disp_alloc(struct avx_disp *m, uint32_t cap) {
    m->keys = map_mmap(cap);
    uint32_t ng = cap >> 3;
    m->disp = (uint8_t *)calloc(ng, 1);
    m->cap   = cap;
    m->mask  = ng - 1;
    m->count = 0;
}

static inline void disp_init(struct avx_disp *m) {
    memset(m, 0, sizeof(*m));
}

static inline void disp_destroy(struct avx_disp *m) {
    if (m->keys) munmap(m->keys, mapsize(m->cap));
    free(m->disp);
}

static inline void disp_prefetch(struct avx_disp *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static inline void disp_prefetch2(struct avx_disp *m, uint64_t key) {
    uint32_t gi = map_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static void disp_grow(struct avx_disp *m);

/* contains: identical to baseline (no change) */
static inline int disp_contains(struct avx_disp *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = map_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle))        return 1;
        if (_mm512_testn_epi64_mask(group, group))          return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* insert: track displacement in disp byte */
static inline int disp_insert(struct avx_disp *m, uint64_t key) {
    if (m->cap == 0) disp_alloc(m, INIT_CAP);
    if (m->count * LOAD_DEN >= m->cap * LOAD_NUM)
        disp_grow(m);

    uint32_t home_gi = map_hash(key) & m->mask;
    uint32_t gi = home_gi;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, _mm512_set1_epi64((long long)key)))
            return 0;
        __mmask8 em = _mm512_testn_epi64_mask(group, group);
        if (em) {
            int slot = __builtin_ctz(em);
            grp[slot] = key;
            m->count++;
            if (gi != home_gi)
                m->disp[gi] |= (uint8_t)(1 << slot);
            return 1;
        }
        gi = (gi + 1) & m->mask;
    }
}

/* delete: use disp to skip all-home groups in backshift */
static inline int disp_delete(struct avx_disp *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    uint32_t gi   = map_hash(key) & m->mask;
    uint32_t mask = m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm    = _mm512_cmpeq_epi64_mask(group, needle);
        __mmask8 empty = _mm512_testn_epi64_mask(group, group);

        if (mm) {
            int del_slot = __builtin_ctz(mm);
            grp[del_slot] = 0;
            /* clear displacement bit for deleted slot */
            m->disp[gi] &= ~(uint8_t)(1 << del_slot);
            m->count--;

            if (empty) return 1;

            uint32_t hole_gi = gi;
            int hole_slot = del_slot;
            uint32_t scan_gi = (gi + 1) & mask;

            for (;;) {
                uint32_t pf_gi = (scan_gi + 2) & mask;
                _mm_prefetch((const char *)(m->keys + (pf_gi << 3)), _MM_HINT_T0);

                uint64_t *scan_grp = m->keys + (scan_gi << 3);
                __m512i scan_group = _mm512_load_si512((__m512i *)scan_grp);
                __mmask8 scan_empty = _mm512_testn_epi64_mask(scan_group, scan_group);

                if (scan_empty == 0xFF) break;

                /* KEY OPTIMIZATION: if no displaced keys in this group,
                 * skip CRC32 hashing entirely */
                uint8_t disp_byte = m->disp[scan_gi];
                __mmask8 displaced = disp_byte & (~scan_empty) & 0xFF;

                if (!displaced) {
                    /* all keys at home → none can move backward */
                    if (scan_empty) break;
                    scan_gi = (scan_gi + 1) & mask;
                    continue;
                }

                /* only hash DISPLACED keys */
                uint64_t cand_keys[8];
                uint32_t cand_homes[8];
                int cand_slots[8];
                int n_cand = 0;

                for (__mmask8 todo = displaced; todo; todo &= todo - 1) {
                    int s = __builtin_ctz(todo);
                    cand_keys[n_cand] = scan_grp[s];
                    cand_slots[n_cand] = s;
                    n_cand++;
                }

                for (int j = 0; j < n_cand; j++)
                    cand_homes[j] = map_hash(cand_keys[j]) & mask;

                int moved = 0;
                for (int j = 0; j < n_cand; j++) {
                    if (((hole_gi - cand_homes[j]) & mask) < ((scan_gi - cand_homes[j]) & mask)) {
                        uint64_t *hole_grp = m->keys + (hole_gi << 3);
                        hole_grp[hole_slot] = cand_keys[j];
                        scan_grp[cand_slots[j]] = 0;

                        /* update displacement bits */
                        m->disp[scan_gi] &= ~(uint8_t)(1 << cand_slots[j]);
                        if (hole_gi != cand_homes[j])
                            m->disp[hole_gi] |= (uint8_t)(1 << hole_slot);
                        /* else: key moved to its home group, bit stays 0 */

                        if (scan_empty) return 1;
                        hole_gi = scan_gi;
                        hole_slot = cand_slots[j];
                        moved = 1;
                        break;
                    }
                }

                if (!moved && scan_empty) break;
                scan_gi = (scan_gi + 1) & mask;
            }
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & mask;
    }
}

static void disp_grow(struct avx_disp *m) {
    uint32_t old_cap = m->cap;
    uint64_t *old_keys = m->keys;
    uint8_t  *old_disp = m->disp;

    disp_alloc(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t i = 0; i < old_cap; i++) {
        uint64_t key = old_keys[i];
        if (!key) continue;
        uint32_t home_gi = map_hash(key) & mask;
        uint32_t gi = home_gi;
        for (;;) {
            uint64_t *grp = m->keys + (gi << 3);
            __mmask8 em = _mm512_testn_epi64_mask(
                _mm512_load_si512((const __m512i *)grp),
                _mm512_load_si512((const __m512i *)grp));
            if (em) {
                int slot = __builtin_ctz(em);
                grp[slot] = key;
                m->count++;
                if (gi != home_gi)
                    m->disp[gi] |= (uint8_t)(1 << slot);
                break;
            }
            gi = (gi + 1) & mask;
        }
    }
    munmap(old_keys, mapsize(old_cap));
    free(old_disp);
}

/* ================================================================
 * Benchmark infrastructure
 * ================================================================ */

struct bench_del_result {
    double del_mops;
    double mixed_mops;
    uint64_t pool_size;
    uint64_t final_live;
    uint64_t n_lookups;
    uint64_t n_inserts;
    uint64_t n_deletes;
    int verified;
};

/* Zipf sampler */
static double *zipf_cdf;
static uint64_t zipf_n;

static void zipf_setup(uint64_t n, double s) {
    zipf_n = n;
    zipf_cdf = (double *)malloc(n * sizeof(double));
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += 1.0 / pow((double)(i + 1), s);
        zipf_cdf[i] = sum;
    }
    for (uint64_t i = 0; i < n; i++) zipf_cdf[i] /= sum;
}

static inline uint64_t zipf_sample(void) {
    double u = (xoshiro256ss() >> 11) * 0x1.0p-53;
    uint64_t lo = 0, hi = zipf_n - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (zipf_cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    return lo + 1;
}

/*
 * Generic benchmark driver — takes function pointers so we can
 * test any variant with the same workload.
 */

typedef void (*init_fn)(void *m);
typedef void (*destroy_fn)(void *m);
typedef int  (*insert_fn)(void *m, uint64_t key);
typedef int  (*contains_fn)(void *m, uint64_t key);
typedef int  (*delete_fn)(void *m, uint64_t key);
typedef void (*prefetch_fn)(void *m, uint64_t key);
typedef void (*prefetch2_fn)(void *m, uint64_t key);
typedef uint32_t (*count_fn)(void *m);

struct variant_ops {
    const char *name;
    size_t struct_size;
    init_fn     init;
    destroy_fn  destroy;
    insert_fn   insert;
    contains_fn contains;
    delete_fn   del;
    prefetch_fn prefetch;
    prefetch2_fn prefetch2;
    count_fn    get_count;
};

/* Wrapper functions to cast void* properly */

/* --- baseline wrappers --- */
static void w_base_init(void *m)     { simd_map64_init((struct simd_map64 *)m); }
static void w_base_destroy(void *m)  { simd_map64_destroy((struct simd_map64 *)m); }
static int  w_base_insert(void *m, uint64_t k) { return simd_map64_insert((struct simd_map64 *)m, k); }
static int  w_base_contains(void *m, uint64_t k) { return simd_map64_contains((struct simd_map64 *)m, k); }
static int  w_base_delete(void *m, uint64_t k) { return simd_map64_delete((struct simd_map64 *)m, k); }
static void w_base_prefetch(void *m, uint64_t k) { simd_map64_prefetch((struct simd_map64 *)m, k); }
static void w_base_prefetch2(void *m, uint64_t k) { simd_map64_prefetch2((struct simd_map64 *)m, k); }
static uint32_t w_base_count(void *m) { return ((struct simd_map64 *)m)->count; }

/* --- tombstone wrappers --- */
static void w_tomb_init(void *m)     { tomb_init((struct avx_tomb *)m); }
static void w_tomb_destroy(void *m)  { tomb_destroy((struct avx_tomb *)m); }
static int  w_tomb_insert(void *m, uint64_t k) { return tomb_insert((struct avx_tomb *)m, k); }
static int  w_tomb_contains(void *m, uint64_t k) { return tomb_contains((struct avx_tomb *)m, k); }
static int  w_tomb_delete(void *m, uint64_t k) { return tomb_delete((struct avx_tomb *)m, k); }
static void w_tomb_prefetch(void *m, uint64_t k) { tomb_prefetch((struct avx_tomb *)m, k); }
static void w_tomb_prefetch2(void *m, uint64_t k) { tomb_prefetch2((struct avx_tomb *)m, k); }
static uint32_t w_tomb_count(void *m) { return ((struct avx_tomb *)m)->count; }

/* --- hybrid wrappers --- */
static void w_hyb_init(void *m)      { hybrid_init((struct avx_hybrid *)m); }
static void w_hyb_destroy(void *m)   { hybrid_destroy((struct avx_hybrid *)m); }
static int  w_hyb_insert(void *m, uint64_t k) { return hybrid_insert((struct avx_hybrid *)m, k); }
static int  w_hyb_contains(void *m, uint64_t k) { return hybrid_contains((struct avx_hybrid *)m, k); }
static int  w_hyb_delete(void *m, uint64_t k) { return hybrid_delete((struct avx_hybrid *)m, k); }
static void w_hyb_prefetch(void *m, uint64_t k) { hybrid_prefetch((struct avx_hybrid *)m, k); }
static void w_hyb_prefetch2(void *m, uint64_t k) { hybrid_prefetch2((struct avx_hybrid *)m, k); }
static uint32_t w_hyb_count(void *m) { return ((struct avx_hybrid *)m)->count; }

/* --- disp bitmap wrappers --- */
static void w_disp_init(void *m)     { disp_init((struct avx_disp *)m); }
static void w_disp_destroy(void *m)  { disp_destroy((struct avx_disp *)m); }
static int  w_disp_insert(void *m, uint64_t k) { return disp_insert((struct avx_disp *)m, k); }
static int  w_disp_contains(void *m, uint64_t k) { return disp_contains((struct avx_disp *)m, k); }
static int  w_disp_delete(void *m, uint64_t k) { return disp_delete((struct avx_disp *)m, k); }
static void w_disp_prefetch(void *m, uint64_t k) { disp_prefetch((struct avx_disp *)m, k); }
static void w_disp_prefetch2(void *m, uint64_t k) { disp_prefetch2((struct avx_disp *)m, k); }
static uint32_t w_disp_count(void *m) { return ((struct avx_disp *)m)->count; }

static struct variant_ops variants[] = {
    { "baseline",   sizeof(struct simd_map64),  w_base_init, w_base_destroy, w_base_insert, w_base_contains, w_base_delete, w_base_prefetch, w_base_prefetch2, w_base_count },
    { "tombstone",  sizeof(struct avx_tomb),   w_tomb_init, w_tomb_destroy, w_tomb_insert, w_tomb_contains, w_tomb_delete, w_tomb_prefetch, w_tomb_prefetch2, w_tomb_count },
    { "hybrid",     sizeof(struct avx_hybrid), w_hyb_init,  w_hyb_destroy,  w_hyb_insert,  w_hyb_contains,  w_hyb_delete,  w_hyb_prefetch,  w_hyb_prefetch2,  w_hyb_count },
    { "disp-bmap",  sizeof(struct avx_disp),   w_disp_init, w_disp_destroy, w_disp_insert, w_disp_contains, w_disp_delete, w_disp_prefetch, w_disp_prefetch2, w_disp_count },
};
#define N_VARIANTS (sizeof(variants)/sizeof(variants[0]))

static struct bench_del_result run_bench(struct variant_ops *v,
                                         uint64_t pool_size, uint64_t n_mixed_ops,
                                         double zipf_s,
                                         int pct_lookup, int pct_insert, int pct_delete)
{
    struct bench_del_result r;
    memset(&r, 0, sizeof(r));
    r.pool_size = pool_size;
    r.verified  = 1;
    int thresh_ins = pct_lookup + pct_insert;
    (void)pct_delete;

    /* save RNG state so all variants get identical workloads */
    uint64_t saved_rng[4];
    memcpy(saved_rng, rng_s, sizeof(rng_s));

    /* generate unique pool */
    char map_buf[256] __attribute__((aligned(64)));
    memset(map_buf, 0, sizeof(map_buf));
    v->init(map_buf);

    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 2; /* avoid 0 (empty) and 1 (tombstone) */
        if (v->insert(map_buf, k))
            pool[gen++] = k;
    }

    /* shuffle */
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }

    /* pure delete: drain entire table */
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < pool_size; i++) {
        if (i + PF_DIST < pool_size)
            v->prefetch2(map_buf, pool[i + PF_DIST]);
        tot += (uint64_t)v->del(map_buf, pool[i]);
    }
    double elapsed = now_sec() - t0;
    r.del_mops = (double)pool_size / elapsed / 1e6;

    if (v->get_count(map_buf) != 0 || tot != pool_size) {
        fprintf(stderr, "[%s] FAIL: delete-all count=%u deleted=%lu\n",
                v->name, v->get_count(map_buf), (unsigned long)tot);
        r.verified = 0;
    }
    v->destroy(map_buf);

    /* mixed workload */
    memset(map_buf, 0, sizeof(map_buf));
    v->init(map_buf);
    uint32_t live  = (uint32_t)(pool_size / 2);
    uint32_t total = (uint32_t)pool_size;
    for (uint32_t i = 0; i < live; i++)
        v->insert(map_buf, pool[i]);

    zipf_setup(live, zipf_s);

    uint64_t *op_keys = (uint64_t *)malloc(n_mixed_ops * sizeof(uint64_t));
    uint8_t  *op_type = (uint8_t *)malloc(n_mixed_ops);

    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
        if (pct < (uint32_t)pct_lookup && live > 0) {
            op_type[i] = 0;
            op_keys[i] = pool[(zipf_sample() - 1) % live];
            r.n_lookups++;
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
            op_keys[i] = pool[di];
            uint64_t tmp = pool[di]; pool[di] = pool[live]; pool[live] = tmp;
            live++;
            r.n_inserts++;
        } else if (live > 2) {
            op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
            op_keys[i] = pool[li];
            live--;
            uint64_t tmp = pool[li]; pool[li] = pool[live]; pool[live] = tmp;
            r.n_deletes++;
        } else {
            op_type[i] = 0;
            op_keys[i] = (live > 0) ? pool[(uint32_t)(xoshiro256ss() >> 32) % live]
                                     : (xoshiro256ss() | 2);
            r.n_lookups++;
        }
    }

    free(zipf_cdf); zipf_cdf = NULL;

    tot = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        if (i + PF_DIST < n_mixed_ops) {
            if (op_type[i + PF_DIST] == 0)
                v->prefetch(map_buf, op_keys[i + PF_DIST]);
            else
                v->prefetch2(map_buf, op_keys[i + PF_DIST]);
        }
        switch (op_type[i]) {
            case 0: tot += (uint64_t)v->contains(map_buf, op_keys[i]); break;
            case 1: tot += (uint64_t)v->insert(map_buf, op_keys[i]); break;
            case 2: tot += (uint64_t)v->del(map_buf, op_keys[i]); break;
        }
    }
    elapsed = now_sec() - t0;
    r.mixed_mops = (double)n_mixed_ops / elapsed / 1e6;
    r.final_live = v->get_count(map_buf);

    /* verify */
    if (v->get_count(map_buf) != live) {
        fprintf(stderr, "[%s] FAIL: mixed count=%u expected=%u\n",
                v->name, v->get_count(map_buf), live);
        r.verified = 0;
    }
    if (r.verified) {
        for (uint32_t i = 0; i < live; i++) {
            if (!v->contains(map_buf, pool[i])) {
                fprintf(stderr, "[%s] FAIL: live pool[%u]=%lu missing\n",
                        v->name, i, (unsigned long)pool[i]);
                r.verified = 0;
                break;
            }
        }
    }
    if (r.verified && total > live) {
        for (uint32_t i = live; i < total; i++) {
            if (v->contains(map_buf, pool[i])) {
                fprintf(stderr, "[%s] FAIL: dead pool[%u]=%lu found\n",
                        v->name, i, (unsigned long)pool[i]);
                r.verified = 0;
                break;
            }
        }
    }

    (void)tot;
    free(op_keys);
    free(op_type);
    v->destroy(map_buf);
    free(pool);
    return r;
}

/* ================================================================
 * Instrumented baseline run — gather statistics
 * ================================================================ */

static void run_instrumented(uint64_t pool_size) {
    memset(&g_stats, 0, sizeof(g_stats));

    struct simd_map64 m;
    simd_map64_init(&m);

    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 2;
        if (simd_map64_insert(&m, k))
            pool[gen++] = k;
    }

    /* shuffle */
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }

    /* delete all — gather stats */
    for (uint64_t i = 0; i < pool_size; i++)
        instrumented_delete(&m, pool[i]);

    printf("=== Backshift instrumentation (pool=%lu) ===\n", (unsigned long)pool_size);
    printf("  total deletes:        %lu\n", (unsigned long)g_stats.total_deletes);
    printf("  no backshift needed:  %lu (%.1f%%)\n",
           (unsigned long)g_stats.no_backshift,
           100.0 * g_stats.no_backshift / g_stats.total_deletes);
    printf("  backshift triggered:  %lu (%.1f%%)\n",
           (unsigned long)g_stats.backshift_triggered,
           100.0 * g_stats.backshift_triggered / g_stats.total_deletes);
    printf("  total groups scanned: %lu (avg %.2f per triggered backshift)\n",
           (unsigned long)g_stats.total_groups_scanned,
           g_stats.backshift_triggered ?
               (double)g_stats.total_groups_scanned / g_stats.backshift_triggered : 0);
    printf("  total keys hashed:    %lu (avg %.2f per group scanned)\n",
           (unsigned long)g_stats.total_keys_hashed,
           g_stats.total_groups_scanned ?
               (double)g_stats.total_keys_hashed / g_stats.total_groups_scanned : 0);
    printf("  keys displaced:       %lu (%.1f%% of hashed)\n",
           (unsigned long)g_stats.total_keys_displaced,
           g_stats.total_keys_hashed ?
               100.0 * g_stats.total_keys_displaced / g_stats.total_keys_hashed : 0);
    printf("  keys moved:           %lu\n", (unsigned long)g_stats.total_keys_moved);
    printf("  chain length distribution:\n");
    for (int i = 0; i <= 16; i++) {
        if (g_stats.chain_hist[i])
            printf("    len %2d: %lu\n", i, (unsigned long)g_stats.chain_hist[i]);
    }
    printf("\n");

    simd_map64_destroy(&m);
    free(pool);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    seed_rng();

    uint64_t pool_size   = 1000000;
    uint64_t n_mixed_ops = 5000000;
    double   zipf_s      = 0.8;

    /* Phase 1: Instrumentation */
    printf("Phase 1: Instrumentation\n");
    printf("========================\n");
    run_instrumented(pool_size);

    /* Phase 2: Benchmark all variants across workload profiles */
    struct { const char *name; int lkp, ins, del; } profiles[] = {
        { "read-heavy",  90,  5,  5 },
        { "balanced",    50, 25, 25 },
        { "churn",       33, 33, 34 },
        { "write-heavy", 10, 50, 40 },
        { "eviction",    20, 10, 70 },
    };
    int n_profiles = sizeof(profiles) / sizeof(profiles[0]);

    printf("Phase 2: Variant comparison (pool=%luK, mixed=%luM)\n",
           (unsigned long)(pool_size/1000), (unsigned long)(n_mixed_ops/1000000));
    printf("==========================================================\n\n");

    /* header */
    printf("%-12s", "profile");
    for (int v = 0; v < (int)N_VARIANTS; v++)
        printf("  %12s", variants[v].name);
    printf("\n");
    for (int i = 0; i < 12 + (int)N_VARIANTS * 14; i++) putchar('-');
    printf("\n");

    /* pure delete */
    printf("%-12s", "pure-del");
    for (int v = 0; v < (int)N_VARIANTS; v++) {
        seed_rng(); /* reset RNG for fair comparison */
        struct bench_del_result r = run_bench(&variants[v],
            pool_size, n_mixed_ops, zipf_s, 50, 25, 25);
        printf("  %9.1f M/s", r.del_mops);
    }
    printf("\n");

    /* mixed profiles */
    for (int p = 0; p < n_profiles; p++) {
        printf("%-12s", profiles[p].name);
        for (int v = 0; v < (int)N_VARIANTS; v++) {
            seed_rng();
            struct bench_del_result r = run_bench(&variants[v],
                pool_size, n_mixed_ops, zipf_s,
                profiles[p].lkp, profiles[p].ins, profiles[p].del);
            char suffix = r.verified ? ' ' : '!';
            printf("  %9.1f%c/s", r.mixed_mops, suffix);
        }
        printf("\n");
    }

    printf("\n(! = verification failed)\n");
    return 0;
}
