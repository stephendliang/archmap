#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>

/* ── Arena bump allocator ────────────────────────────────────────────── */

#define ARENA_DEFAULT_BLOCK (2u << 20)  /* 2 MiB */
#define ARENA_ALIGN 16

struct arena_block {
    struct arena_block *next;
    size_t size;
    size_t used;
};

struct arena {
    struct arena_block *head;
    size_t default_block;
};

void arena_init(struct arena *a, size_t block_size);
void *arena_alloc(struct arena *a, size_t n);
char *arena_strdup(struct arena *a, const char *s);
void *arena_realloc(struct arena *a, void *old_ptr,
                    size_t old_size, size_t new_size);
size_t arena_save(struct arena *a);
void arena_reset(struct arena *a, size_t mark);
void arena_destroy(struct arena *a);

/* ── String intern table ─────────────────────────────────────────────── */

struct intern_slot {
    uint64_t hash;
    const char *str;
};

struct intern_table {
    struct intern_slot *slots;
    uint32_t cap, count;
    struct arena arena;
};

void intern_init(struct intern_table *t, uint32_t initial_cap);
const char *intern_str(struct intern_table *t, const char *s);
void intern_destroy(struct intern_table *t);

/* ── OOM-aborting allocator wrappers ──────────────────────────────────── */

void *xmalloc(size_t n);
void *xcalloc(size_t count, size_t size);
void *xrealloc(void *ptr, size_t n);

#endif /* ARENA_H */
