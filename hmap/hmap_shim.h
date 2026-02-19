/*
 * hmap_shim.h — Verstable-backed shim for the SAHA string-set interface.
 *
 * Mirrors SAHA's 4-tier structure with 4 separate verstable instances,
 * each keyed by a fixed-width value type for short keys (inline, no strcmp)
 * and a pointer type for long keys (>24 bytes, rare).
 *
 * Provides: saha_init, saha_destroy, saha_insert, saha_contains.
 */
#ifndef HMAP_SHIM_H
#define HMAP_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <nmmintrin.h> /* SSE4.2 CRC32 */

/* ── Key types for tiers 2 and 3 ─────────────────────────────────── */

struct saha_key16 { uint64_t lo, hi; };
struct saha_key24 { uint64_t a, b, c; };

/* ── Per-tier CRC32C hash functions (SSE4.2) ─────────────────────── */

static inline uint64_t saha_t1_hash(uint64_t k)
{
    uint64_t h = _mm_crc32_u64(0, k);
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

static inline uint64_t saha_t2_hash(struct saha_key16 k)
{
    uint64_t h1 = _mm_crc32_u64(0, k.lo);
    uint64_t h2 = _mm_crc32_u64(0x9E3779B9, k.hi);
    uint64_t h  = h1 ^ (h2 << 32);
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

static inline uint64_t saha_t3_hash(struct saha_key24 k)
{
    uint64_t h1 = _mm_crc32_u64(0, k.a);
    uint64_t h2 = _mm_crc32_u64(0x517CC1B7, k.b);
    uint64_t h3 = _mm_crc32_u64(0x9E3779B9, k.c);
    uint64_t h  = h1 ^ (h2 << 21) ^ (h3 << 42);
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

static inline uint64_t saha_t4_hash(const char *key)
{
    uint64_t h   = 0;
    size_t   len = strlen(key);
    while (len >= 8) {
        uint64_t v;
        memcpy(&v, key, 8);
        h = _mm_crc32_u64(h, v);
        key += 8;
        len -= 8;
    }
    if (len) {
        uint64_t v = 0;
        memcpy(&v, key, len);
        h = _mm_crc32_u64(h, v);
    }
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

/* ── Per-tier comparators ────────────────────────────────────────── */

static inline bool saha_t2_cmpr(struct saha_key16 a, struct saha_key16 b)
{
    return a.lo == b.lo && a.hi == b.hi;
}

static inline bool saha_t3_cmpr(struct saha_key24 a, struct saha_key24 b)
{
    return a.a == b.a && a.b == b.b && a.c == b.c;
}

/* ── Tier 4 key destructor (strdup'd strings) ────────────────────── */

static inline void saha_free_key(const char *key)
{
    free((void *)key);
}

/* ── 4 verstable instantiations ──────────────────────────────────── */

#define NAME     saha_t1
#define KEY_TY   uint64_t
#define HASH_FN  saha_t1_hash
#define CMPR_FN  vt_cmpr_integer
#include "verstable.h"

#define NAME     saha_t2
#define KEY_TY   struct saha_key16
#define HASH_FN  saha_t2_hash
#define CMPR_FN  saha_t2_cmpr
#include "verstable.h"

#define NAME     saha_t3
#define KEY_TY   struct saha_key24
#define HASH_FN  saha_t3_hash
#define CMPR_FN  saha_t3_cmpr
#include "verstable.h"

#define NAME        saha_t4
#define KEY_TY      const char *
#define HASH_FN     saha_t4_hash
#define CMPR_FN     vt_cmpr_string
#define KEY_DTOR_FN saha_free_key
#include "verstable.h"

/* ── Composite container ─────────────────────────────────────────── */

typedef struct saha {
    saha_t1 t1;
    saha_t2 t2;
    saha_t3 t3;
    saha_t4 t4;
} saha;

static inline void saha_init(saha *s)
{
    saha_t1_init(&s->t1);
    saha_t2_init(&s->t2);
    saha_t3_init(&s->t3);
    saha_t4_init(&s->t4);
}

static inline void saha_destroy(saha *s)
{
    saha_t1_cleanup(&s->t1);
    saha_t2_cleanup(&s->t2);
    saha_t3_cleanup(&s->t3);
    saha_t4_cleanup(&s->t4);
}

static inline int saha_insert(saha *s, const char *key, size_t len)
{
    if (len <= 8) {
        uint64_t k = 0;
        memcpy(&k, key, len);
        size_t before = saha_t1_size(&s->t1);
        saha_t1_get_or_insert(&s->t1, k);
        return saha_t1_size(&s->t1) != before;
    } else if (len <= 16) {
        struct saha_key16 k = {0};
        memcpy(&k.lo, key, 8);
        memcpy(&k.hi, key + 8, len - 8);
        size_t before = saha_t2_size(&s->t2);
        saha_t2_get_or_insert(&s->t2, k);
        return saha_t2_size(&s->t2) != before;
    } else if (len <= 24) {
        struct saha_key24 k = {0};
        memcpy(&k.a, key, 8);
        memcpy(&k.b, key + 8, 8);
        memcpy(&k.c, key + 16, len - 16);
        size_t before = saha_t3_size(&s->t3);
        saha_t3_get_or_insert(&s->t3, k);
        return saha_t3_size(&s->t3) != before;
    } else {
        if (!saha_t4_is_end(saha_t4_get(&s->t4, key)))
            return 0;
        saha_t4_insert(&s->t4, strdup(key));
        return 1;
    }
}

static inline int saha_contains(saha *s, const char *key, size_t len)
{
    if (len <= 8) {
        uint64_t k = 0;
        memcpy(&k, key, len);
        return !saha_t1_is_end(saha_t1_get(&s->t1, k));
    } else if (len <= 16) {
        struct saha_key16 k = {0};
        memcpy(&k.lo, key, 8);
        memcpy(&k.hi, key + 8, len - 8);
        return !saha_t2_is_end(saha_t2_get(&s->t2, k));
    } else if (len <= 24) {
        struct saha_key24 k = {0};
        memcpy(&k.a, key, 8);
        memcpy(&k.b, key + 8, 8);
        memcpy(&k.c, key + 16, len - 16);
        return !saha_t3_is_end(saha_t3_get(&s->t3, k));
    } else {
        return !saha_t4_is_end(saha_t4_get(&s->t4, key));
    }
}

#endif /* HMAP_SHIM_H */
