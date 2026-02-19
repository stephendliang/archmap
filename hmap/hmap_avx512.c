/*
 * SAHA: SIMD-Accelerated Hash Array — implementation
 */
#include "hmap_avx512.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SAHA_INIT_CAP   64   /* one group */
#define SAHA_ARENA_INIT 4096
#define SAHA_LOAD_NUM   7    /* grow when count*8 >= cap*7 (87.5%) */
#define SAHA_LOAD_DEN   8

/* ================================================================
 * Hash function (wyhash core)
 * ================================================================ */

static inline uint64_t _wymix(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

static inline uint64_t _wyr8(const void *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

static inline uint64_t _wyr4(const void *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

static inline uint64_t _wyr3(const void *p, size_t k) {
    const uint8_t *b = (const uint8_t *)p;
    return ((uint64_t)b[0] << 16) | ((uint64_t)b[k >> 1] << 8) | b[k - 1];
}

static uint64_t saha_hash(const void *key, size_t len) {
    const uint64_t s0 = 0xa0761d6478bd642fULL;
    const uint64_t s1 = 0xe7037ed1a0b428dbULL;
    const uint64_t s2 = 0x8ebc6af09c88c6e3ULL;
    const uint64_t s3 = 0x589965cc75374cc3ULL;
    const uint8_t *p = (const uint8_t *)key;
    uint64_t seed = s0, a, b;

    if (__builtin_expect(len <= 16, 1)) {
        if (__builtin_expect(len >= 4, 1)) {
            a = (_wyr4(p) << 32) | _wyr4(p + ((len >> 3) << 2));
            b = (_wyr4(p + len - 4) << 32) |
                _wyr4(p + len - 4 - ((len >> 3) << 2));
        } else if (__builtin_expect(len > 0, 1)) {
            a = _wyr3(p, len);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                seed = _wymix(_wyr8(p)      ^ s1, _wyr8(p + 8)  ^ seed);
                see1 = _wymix(_wyr8(p + 16) ^ s2, _wyr8(p + 24) ^ see1);
                see2 = _wymix(_wyr8(p + 32) ^ s3, _wyr8(p + 40) ^ see2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = _wymix(_wyr8(p) ^ s1, _wyr8(p + 8) ^ seed);
            i -= 16; p += 16;
        }
        a = _wyr8(p + i - 16);
        b = _wyr8(p + i - 8);
    }
    return _wymix(s1 ^ len, _wymix(a ^ s1, b ^ seed));
}

/* ================================================================
 * SIMD helpers
 * ================================================================ */

static inline uint8_t saha_h2(uint64_t hash) {
    return (uint8_t)((hash >> 57) | 0x80);
}

static inline __mmask64 saha_match(const uint8_t *meta, uint8_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi8((char)h2);
    return _mm512_cmpeq_epi8_mask(group, needle);
}

static inline __mmask64 saha_empty(const uint8_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_cmpeq_epi8_mask(group, _mm512_setzero_si512());
}

/* ================================================================
 * Key packing
 * ================================================================ */

static inline uint64_t saha_pack8(const char *key, size_t len) {
    uint64_t v = 0;
    memcpy(&v, key, len);
    return v;
}

static inline struct saha_key16 saha_pack16(const char *key, size_t len) {
    struct saha_key16 k = {0, 0};
    memcpy(&k.lo, key, 8);
    memcpy(&k.hi, key + 8, len - 8);
    return k;
}

static inline struct saha_key24 saha_pack24(const char *key, size_t len) {
    struct saha_key24 k = {0, 0, 0};
    memcpy(&k.a, key, 8);
    memcpy(&k.b, key + 8, 8);
    memcpy(&k.c, key + 16, len - 16);
    return k;
}

/* ================================================================
 * Tier 1: keys <= 8 bytes (inline u64)
 * ================================================================ */

static void saha_t1_alloc(struct saha_tier1 *t, uint32_t cap) {
    t->meta = (uint8_t *)aligned_alloc(64, cap);
    memset(t->meta, 0, cap);
    t->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    t->cap  = cap;
    t->count = 0;
}

static void saha_t1_grow(struct saha_tier1 *t) {
    uint32_t old_cap = t->cap;
    uint8_t  *old_meta = t->meta;
    uint64_t *old_keys = t->keys;

    saha_t1_alloc(t, old_cap * 2);
    uint32_t ng = t->cap / 64;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!(old_meta[i] & 0x80)) continue;
        uint64_t packed = old_keys[i];
        uint64_t h = saha_hash(&packed, 8);
        uint8_t  h2 = saha_h2(h);
        uint32_t gi = (uint32_t)h & (ng - 1);
        for (;;) {
            __mmask64 em = saha_empty(t->meta + gi * 64);
            if (em) {
                int pos = __builtin_ctzll(em);
                t->meta[gi * 64 + pos] = h2;
                t->keys[gi * 64 + pos] = packed;
                t->count++;
                break;
            }
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

int saha_t1_insert(struct saha_tier1 *t, const char *key, size_t len) {
    if (t->cap == 0) saha_t1_alloc(t, SAHA_INIT_CAP);
    if (t->count * SAHA_LOAD_DEN >= t->cap * SAHA_LOAD_NUM)
        saha_t1_grow(t);

    uint64_t packed = saha_pack8(key, len);
    uint64_t h  = saha_hash(&packed, 8);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t  *meta = t->meta + gi * 64;
        uint64_t *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos] == packed) return 0;
            m &= m - 1;
        }

        __mmask64 em = saha_empty(meta);
        if (em) {
            int pos = __builtin_ctzll(em);
            meta[pos] = h2;
            keys[pos] = packed;
            t->count++;
            return 1;
        }
        gi = (gi + 1) & (ng - 1);
    }
}

int saha_t1_contains(struct saha_tier1 *t, const char *key, size_t len) {
    if (t->cap == 0) return 0;

    uint64_t packed = saha_pack8(key, len);
    uint64_t h  = saha_hash(&packed, 8);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t  *meta = t->meta + gi * 64;
        uint64_t *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos] == packed) return 1;
            m &= m - 1;
        }
        if (saha_empty(meta)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

/* ================================================================
 * Tier 2: keys 9-16 bytes (inline 2×u64)
 * ================================================================ */

static void saha_t2_alloc(struct saha_tier2 *t, uint32_t cap) {
    t->meta = (uint8_t *)aligned_alloc(64, cap);
    memset(t->meta, 0, cap);
    t->keys = (struct saha_key16 *)malloc(cap * sizeof(struct saha_key16));
    t->cap  = cap;
    t->count = 0;
}

static void saha_t2_grow(struct saha_tier2 *t) {
    uint32_t old_cap = t->cap;
    uint8_t *old_meta = t->meta;
    struct saha_key16 *old_keys = t->keys;

    saha_t2_alloc(t, old_cap * 2);
    uint32_t ng = t->cap / 64;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!(old_meta[i] & 0x80)) continue;
        struct saha_key16 pk = old_keys[i];
        uint64_t h  = saha_hash(&pk, 16);
        uint8_t  h2 = saha_h2(h);
        uint32_t gi = (uint32_t)h & (ng - 1);
        for (;;) {
            __mmask64 em = saha_empty(t->meta + gi * 64);
            if (em) {
                int pos = __builtin_ctzll(em);
                t->meta[gi * 64 + pos] = h2;
                t->keys[gi * 64 + pos] = pk;
                t->count++;
                break;
            }
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

int saha_t2_insert(struct saha_tier2 *t, const char *key, size_t len) {
    if (t->cap == 0) saha_t2_alloc(t, SAHA_INIT_CAP);
    if (t->count * SAHA_LOAD_DEN >= t->cap * SAHA_LOAD_NUM)
        saha_t2_grow(t);

    struct saha_key16 pk = saha_pack16(key, len);
    uint64_t h  = saha_hash(&pk, 16);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_key16 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].lo == pk.lo && keys[pos].hi == pk.hi) return 0;
            m &= m - 1;
        }

        __mmask64 em = saha_empty(meta);
        if (em) {
            int pos = __builtin_ctzll(em);
            meta[pos] = h2;
            keys[pos] = pk;
            t->count++;
            return 1;
        }
        gi = (gi + 1) & (ng - 1);
    }
}

int saha_t2_contains(struct saha_tier2 *t, const char *key, size_t len) {
    if (t->cap == 0) return 0;

    struct saha_key16 pk = saha_pack16(key, len);
    uint64_t h  = saha_hash(&pk, 16);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_key16 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].lo == pk.lo && keys[pos].hi == pk.hi) return 1;
            m &= m - 1;
        }
        if (saha_empty(meta)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

/* ================================================================
 * Tier 3: keys 17-24 bytes (inline 3×u64)
 * ================================================================ */

static void saha_t3_alloc(struct saha_tier3 *t, uint32_t cap) {
    t->meta = (uint8_t *)aligned_alloc(64, cap);
    memset(t->meta, 0, cap);
    t->keys = (struct saha_key24 *)malloc(cap * sizeof(struct saha_key24));
    t->cap  = cap;
    t->count = 0;
}

static void saha_t3_grow(struct saha_tier3 *t) {
    uint32_t old_cap = t->cap;
    uint8_t *old_meta = t->meta;
    struct saha_key24 *old_keys = t->keys;

    saha_t3_alloc(t, old_cap * 2);
    uint32_t ng = t->cap / 64;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!(old_meta[i] & 0x80)) continue;
        struct saha_key24 pk = old_keys[i];
        uint64_t h  = saha_hash(&pk, 24);
        uint8_t  h2 = saha_h2(h);
        uint32_t gi = (uint32_t)h & (ng - 1);
        for (;;) {
            __mmask64 em = saha_empty(t->meta + gi * 64);
            if (em) {
                int pos = __builtin_ctzll(em);
                t->meta[gi * 64 + pos] = h2;
                t->keys[gi * 64 + pos] = pk;
                t->count++;
                break;
            }
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

int saha_t3_insert(struct saha_tier3 *t, const char *key, size_t len) {
    if (t->cap == 0) saha_t3_alloc(t, SAHA_INIT_CAP);
    if (t->count * SAHA_LOAD_DEN >= t->cap * SAHA_LOAD_NUM)
        saha_t3_grow(t);

    struct saha_key24 pk = saha_pack24(key, len);
    uint64_t h  = saha_hash(&pk, 24);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_key24 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].a == pk.a && keys[pos].b == pk.b &&
                keys[pos].c == pk.c) return 0;
            m &= m - 1;
        }

        __mmask64 em = saha_empty(meta);
        if (em) {
            int pos = __builtin_ctzll(em);
            meta[pos] = h2;
            keys[pos] = pk;
            t->count++;
            return 1;
        }
        gi = (gi + 1) & (ng - 1);
    }
}

int saha_t3_contains(struct saha_tier3 *t, const char *key, size_t len) {
    if (t->cap == 0) return 0;

    struct saha_key24 pk = saha_pack24(key, len);
    uint64_t h  = saha_hash(&pk, 24);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_key24 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].a == pk.a && keys[pos].b == pk.b &&
                keys[pos].c == pk.c) return 1;
            m &= m - 1;
        }
        if (saha_empty(meta)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

/* ================================================================
 * Tier 4: keys > 24 bytes (hash + arena offset)
 * ================================================================ */

static void saha_t4_alloc_tables(struct saha_tier4 *t, uint32_t cap) {
    t->meta = (uint8_t *)aligned_alloc(64, cap);
    memset(t->meta, 0, cap);
    t->keys = (struct saha_slot4 *)malloc(cap * sizeof(struct saha_slot4));
    t->cap  = cap;
    t->count = 0;
}

static uint32_t saha_arena_store(struct saha_tier4 *t,
                                  const char *key, size_t len) {
    uint32_t needed = (uint32_t)(len + 1);
    if (t->arena_used + needed > t->arena_cap) {
        uint32_t nc = t->arena_cap * 2;
        while (nc < t->arena_used + needed) nc *= 2;
        t->arena = (char *)realloc(t->arena, nc);
        t->arena_cap = nc;
    }
    uint32_t off = t->arena_used;
    memcpy(t->arena + off, key, len);
    t->arena[off + len] = '\0';
    t->arena_used += needed;
    return off;
}

static void saha_t4_grow(struct saha_tier4 *t) {
    uint32_t old_cap = t->cap;
    uint8_t *old_meta = t->meta;
    struct saha_slot4 *old_keys = t->keys;

    saha_t4_alloc_tables(t, old_cap * 2);
    uint32_t ng = t->cap / 64;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!(old_meta[i] & 0x80)) continue;
        struct saha_slot4 slot = old_keys[i];
        uint8_t  h2 = saha_h2(slot.hash);
        uint32_t gi = (uint32_t)slot.hash & (ng - 1);
        for (;;) {
            __mmask64 em = saha_empty(t->meta + gi * 64);
            if (em) {
                int pos = __builtin_ctzll(em);
                t->meta[gi * 64 + pos] = h2;
                t->keys[gi * 64 + pos] = slot;
                t->count++;
                break;
            }
            gi = (gi + 1) & (ng - 1);
        }
    }
    free(old_meta);
    free(old_keys);
}

int saha_t4_insert(struct saha_tier4 *t, const char *key, size_t len) {
    if (t->cap == 0) {
        saha_t4_alloc_tables(t, SAHA_INIT_CAP);
        t->arena = (char *)malloc(SAHA_ARENA_INIT);
        t->arena_cap = SAHA_ARENA_INIT;
        t->arena_used = 0;
    }
    if (t->count * SAHA_LOAD_DEN >= t->cap * SAHA_LOAD_NUM)
        saha_t4_grow(t);

    uint64_t h  = saha_hash(key, len);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_slot4 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].hash == h && keys[pos].len == (uint32_t)len &&
                memcmp(t->arena + keys[pos].off, key, len) == 0)
                return 0;
            m &= m - 1;
        }

        __mmask64 em = saha_empty(meta);
        if (em) {
            int pos = __builtin_ctzll(em);
            uint32_t off = saha_arena_store(t, key, len);
            meta[pos] = h2;
            keys[pos].hash = h;
            keys[pos].off  = off;
            keys[pos].len  = (uint32_t)len;
            t->count++;
            return 1;
        }
        gi = (gi + 1) & (ng - 1);
    }
}

int saha_t4_contains(struct saha_tier4 *t, const char *key, size_t len) {
    if (t->cap == 0) return 0;

    uint64_t h  = saha_hash(key, len);
    uint8_t  h2 = saha_h2(h);
    uint32_t ng = t->cap / 64;
    uint32_t gi = (uint32_t)h & (ng - 1);

    for (;;) {
        uint8_t *meta = t->meta + gi * 64;
        struct saha_slot4 *keys = t->keys + gi * 64;

        __mmask64 m = saha_match(meta, h2);
        while (m) {
            int pos = __builtin_ctzll(m);
            if (keys[pos].hash == h && keys[pos].len == (uint32_t)len &&
                memcmp(t->arena + keys[pos].off, key, len) == 0)
                return 1;
            m &= m - 1;
        }
        if (saha_empty(meta)) return 0;
        gi = (gi + 1) & (ng - 1);
    }
}

/* ================================================================
 * Top-level API
 * ================================================================ */

void saha_init(saha *s) {
    memset(s, 0, sizeof(*s));
}

void saha_destroy(saha *s) {
    free(s->t1.meta); free(s->t1.keys);
    free(s->t2.meta); free(s->t2.keys);
    free(s->t3.meta); free(s->t3.keys);
    free(s->t4.meta); free(s->t4.keys);
    free(s->t4.arena);
}

void saha_dump_stats(const saha *s) {
    uint32_t total = s->t1.count + s->t2.count + s->t3.count + s->t4.count;
    size_t mem = 0;
    if (s->t1.cap) mem += s->t1.cap + s->t1.cap * 8;
    if (s->t2.cap) mem += s->t2.cap + s->t2.cap * 16;
    if (s->t3.cap) mem += s->t3.cap + s->t3.cap * 24;
    if (s->t4.cap) mem += s->t4.cap + s->t4.cap * 16;
    mem += s->t4.arena_cap;
    fprintf(stderr, "saha: %u keys, %.1f KiB\n", total, mem / 1024.0);
    fprintf(stderr, "  t1 (<=8B):  %4u keys, cap %u\n", s->t1.count, s->t1.cap);
    fprintf(stderr, "  t2 (<=16B): %4u keys, cap %u\n", s->t2.count, s->t2.cap);
    fprintf(stderr, "  t3 (<=24B): %4u keys, cap %u\n", s->t3.count, s->t3.cap);
    fprintf(stderr, "  t4 (>24B):  %4u keys, cap %u, arena %u/%u\n",
            s->t4.count, s->t4.cap, s->t4.arena_used, s->t4.arena_cap);
}
