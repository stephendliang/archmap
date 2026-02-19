/* arena.c — Arena bump allocator + string intern table + OOM wrappers */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "arena.h"

/* ── Arena bump allocator ────────────────────────────────────────────── */

void arena_init(struct arena *a, size_t block_size) {
    a->head = NULL;
    a->default_block = block_size;
}

static struct arena_block *arena_new_block(size_t min_size,
                                            size_t default_size) {
    size_t size = min_size > default_size ? min_size : default_size;
    size = (size + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) madvise(p, size, MADV_HUGEPAGE);
    }
    if (p == MAP_FAILED) return NULL;
    struct arena_block *b = (struct arena_block *)p;
    b->next = NULL;
    b->size = size;
    b->used = (sizeof(struct arena_block) + ARENA_ALIGN - 1)
              & ~((size_t)ARENA_ALIGN - 1);
    return b;
}

void *arena_alloc(struct arena *a, size_t n) {
    n = (n + ARENA_ALIGN - 1) & ~((size_t)ARENA_ALIGN - 1);
    struct arena_block *b = a->head;
    if (b && b->used + n <= b->size) {
        void *p = (char *)b + b->used;
        b->used += n;
        return p;
    }
    size_t hdr = (sizeof(struct arena_block) + ARENA_ALIGN - 1)
                 & ~((size_t)ARENA_ALIGN - 1);
    b = arena_new_block(n + hdr, a->default_block);
    if (!b) { fprintf(stderr, "arena_alloc: mmap failed\n"); abort(); }
    b->next = a->head;
    a->head = b;
    void *p = (char *)b + b->used;
    b->used += n;
    return p;
}

char *arena_strdup(struct arena *a, const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *p = arena_alloc(a, len);
    memcpy(p, s, len);
    return p;
}

void *arena_realloc(struct arena *a, void *old_ptr,
                           size_t old_size, size_t new_size) {
    if (!old_ptr) return arena_alloc(a, new_size);
    size_t old_al = (old_size + ARENA_ALIGN - 1) & ~((size_t)ARENA_ALIGN - 1);
    size_t new_al = (new_size + ARENA_ALIGN - 1) & ~((size_t)ARENA_ALIGN - 1);
    struct arena_block *b = a->head;
    if (b && (char *)old_ptr + old_al == (char *)b + b->used
          && b->used - old_al + new_al <= b->size) {
        b->used = b->used - old_al + new_al;
        return old_ptr;
    }
    void *p = arena_alloc(a, new_size);
    memcpy(p, old_ptr, old_size);
    return p;
}

size_t arena_save(struct arena *a) {
    return a->head ? a->head->used : 0;
}

void arena_reset(struct arena *a, size_t mark) {
    if (a->head) a->head->used = mark;
}

void arena_destroy(struct arena *a) {
    struct arena_block *b = a->head;
    while (b) {
        struct arena_block *next = b->next;
        munmap(b, b->size);
        b = next;
    }
    a->head = NULL;
}

/* ── String intern table ─────────────────────────────────────────────── */

/* wyhash core */

static inline uint64_t arena__wymix(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

static inline uint64_t arena__wyr8(const void *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

static inline uint64_t arena__wyr4(const void *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

static inline uint64_t arena__wyr3(const void *p, size_t k) {
    const uint8_t *b = (const uint8_t *)p;
    return ((uint64_t)b[0] << 16) | ((uint64_t)b[k >> 1] << 8) | b[k - 1];
}

static uint64_t intern_hash(const void *key, size_t len) {
    const uint64_t s0 = 0xa0761d6478bd642fULL;
    const uint64_t s1 = 0xe7037ed1a0b428dbULL;
    const uint64_t s2 = 0x8ebc6af09c88c6e3ULL;
    const uint64_t s3 = 0x589965cc75374cc3ULL;
    const uint8_t *p = (const uint8_t *)key;
    uint64_t seed = s0, a, b;

    if (__builtin_expect(len <= 16, 1)) {
        if (__builtin_expect(len >= 4, 1)) {
            a = (arena__wyr4(p) << 32) | arena__wyr4(p + ((len >> 3) << 2));
            b = (arena__wyr4(p + len - 4) << 32) |
                arena__wyr4(p + len - 4 - ((len >> 3) << 2));
        } else if (__builtin_expect(len > 0, 1)) {
            a = arena__wyr3(p, len);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                seed = arena__wymix(arena__wyr8(p)      ^ s1,
                                 arena__wyr8(p + 8)  ^ seed);
                see1 = arena__wymix(arena__wyr8(p + 16) ^ s2,
                                 arena__wyr8(p + 24) ^ see1);
                see2 = arena__wymix(arena__wyr8(p + 32) ^ s3,
                                 arena__wyr8(p + 40) ^ see2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = arena__wymix(arena__wyr8(p) ^ s1, arena__wyr8(p + 8) ^ seed);
            i -= 16; p += 16;
        }
        a = arena__wyr8(p + i - 16);
        b = arena__wyr8(p + i - 8);
    }
    return arena__wymix(s1 ^ len, arena__wymix(a ^ s1, b ^ seed));
}

void intern_init(struct intern_table *t, uint32_t initial_cap) {
    t->cap = initial_cap;
    t->count = 0;
    t->slots = calloc((size_t)initial_cap, sizeof(*t->slots));
    if (!t->slots) { fprintf(stderr, "intern_init: calloc failed\n"); abort(); }
    arena_init(&t->arena, ARENA_DEFAULT_BLOCK);
}

static void intern_grow(struct intern_table *t) {
    uint32_t old_cap = t->cap;
    struct intern_slot *old = t->slots;
    t->cap *= 2;
    t->slots = calloc((size_t)t->cap, sizeof(*t->slots));
    if (!t->slots) { fprintf(stderr, "intern_grow: calloc failed\n"); abort(); }
    for (uint32_t i = 0; i < old_cap; i++) {
        if (!old[i].str) continue;
        uint32_t idx = (uint32_t)(old[i].hash & (t->cap - 1));
        while (t->slots[idx].str)
            idx = (idx + 1) & (t->cap - 1);
        t->slots[idx] = old[i];
    }
    free(old);
}

const char *intern_str(struct intern_table *t, const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    uint64_t h = intern_hash(s, len);
    uint32_t idx = (uint32_t)(h & (t->cap - 1));
    while (t->slots[idx].str) {
        if (t->slots[idx].hash == h && strcmp(t->slots[idx].str, s) == 0)
            return t->slots[idx].str;
        idx = (idx + 1) & (t->cap - 1);
    }
    if (t->count * 4 >= t->cap * 3) {  /* 75% load */
        intern_grow(t);
        idx = (uint32_t)(h & (t->cap - 1));
        while (t->slots[idx].str)
            idx = (idx + 1) & (t->cap - 1);
    }
    const char *interned = arena_strdup(&t->arena, s);
    t->slots[idx].hash = h;
    t->slots[idx].str = interned;
    t->count++;
    return interned;
}

void intern_destroy(struct intern_table *t) {
    free(t->slots);
    t->slots = NULL;
    arena_destroy(&t->arena);
    t->count = t->cap = 0;
}

/* ── OOM-aborting allocator wrappers ──────────────────────────────────── */

void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) { fprintf(stderr, "out of memory (malloc %zu)\n", n); abort(); }
    return p;
}
void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p && count && size) { fprintf(stderr, "out of memory (calloc %zu)\n", count * size); abort(); }
    return p;
}
void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p && n) { fprintf(stderr, "out of memory (realloc %zu)\n", n); abort(); }
    return p;
}
