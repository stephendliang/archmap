/*
 * SAHA: SIMD-Accelerated Hash Array (AVX-512)
 *
 * 4-tier open-addressing hash set with 64-slot groups probed via AVX-512.
 * Tier selection by key length: <=8, <=16, <=24, >24 bytes.
 */
#ifndef HMAP_AVX512_H
#define HMAP_AVX512_H

#include <stdint.h>
#include <stddef.h>

/* --- Key types for inline storage --- */

struct saha_key16 { uint64_t lo, hi; };
struct saha_key24 { uint64_t a, b, c; };
struct saha_slot4 { uint64_t hash; uint32_t off; uint32_t len; };

/* --- Per-tier table structures --- */

struct saha_tier1 {
    uint8_t  *meta;     /* h2 hash bytes, 64-aligned */
    uint64_t *keys;     /* zero-padded key as u64 */
    uint32_t count;
    uint32_t cap;       /* always multiple of 64 */
};

struct saha_tier2 {
    uint8_t *meta;
    struct saha_key16 *keys;
    uint32_t count;
    uint32_t cap;
};

struct saha_tier3 {
    uint8_t *meta;
    struct saha_key24 *keys;
    uint32_t count;
    uint32_t cap;
};

struct saha_tier4 {
    uint8_t *meta;
    struct saha_slot4 *keys;
    uint32_t count;
    uint32_t cap;
    char    *arena;
    uint32_t arena_used;
    uint32_t arena_cap;
};

/* --- Top-level container --- */

typedef struct saha {
    struct saha_tier1 t1;
    struct saha_tier2 t2;
    struct saha_tier3 t3;
    struct saha_tier4 t4;
} saha;

/* --- Per-tier functions (defined in hmap_avx512.c) --- */

int saha_t1_insert(struct saha_tier1 *t, const char *key, size_t len);
int saha_t1_contains(struct saha_tier1 *t, const char *key, size_t len);
int saha_t2_insert(struct saha_tier2 *t, const char *key, size_t len);
int saha_t2_contains(struct saha_tier2 *t, const char *key, size_t len);
int saha_t3_insert(struct saha_tier3 *t, const char *key, size_t len);
int saha_t3_contains(struct saha_tier3 *t, const char *key, size_t len);
int saha_t4_insert(struct saha_tier4 *t, const char *key, size_t len);
int saha_t4_contains(struct saha_tier4 *t, const char *key, size_t len);

void saha_init(saha *s);
void saha_destroy(saha *s);
void saha_dump_stats(const saha *s);

/* --- Inline tier dispatch --- */

static inline int saha_insert(saha *s, const char *key, size_t len) {
    if (len <= 8)  return saha_t1_insert(&s->t1, key, len);
    if (len <= 16) return saha_t2_insert(&s->t2, key, len);
    if (len <= 24) return saha_t3_insert(&s->t3, key, len);
    return saha_t4_insert(&s->t4, key, len);
}

static inline int saha_contains(saha *s, const char *key, size_t len) {
    if (len <= 8)  return saha_t1_contains(&s->t1, key, len);
    if (len <= 16) return saha_t2_contains(&s->t2, key, len);
    if (len <= 24) return saha_t3_contains(&s->t3, key, len);
    return saha_t4_contains(&s->t4, key, len);
}

#endif /* HMAP_AVX512_H */
