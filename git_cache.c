/* git_cache.c — libgit2-based incremental cache for archmap
 *
 * Keys file parse results by git blob OID (SHA-1 of content).
 * Unchanged files get identical OIDs across runs, so we skip tree-sitter.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#include <git2.h>

#include "git_cache.h"
#include "perf_analysis.h"

/* ── Binary format constants ─────────────────────────────────────────── */

#define CACHE_MAGIC "AMAP"
#define CACHE_VERSION 1
#define HEADER_SIZE 32

/* ── FNV-1a hash ─────────────────────────────────────────────────────── */

static uint64_t fnv1a(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ── Internal types ──────────────────────────────────────────────────── */

struct cache_entry {
    git_oid blob_oid;
    char *abs_path;
    struct file_entry fe;   /* deep copy (abs_path inside points to own alloc) */
    char **tag_names;
    int n_tags;
};

struct archmap_cache {
    git_repository *repo;
    char cache_path[PATH_MAX];
    uint64_t opts_hash;

    struct cache_entry *entries;
    int n_entries, cap_entries;

    struct cache_entry *fresh;
    int n_fresh, cap_fresh;

    int dirty;
};

/* ── Deep copy helpers ───────────────────────────────────────────────── */

static struct symbol deep_copy_symbol(const struct symbol *s) {
    struct symbol out;
    memset(&out, 0, sizeof(out));
    out.text = s->text ? strdup(s->text) : NULL;
    out.cap_idx = s->cap_idx;
    out.start_line = s->start_line;
    out.end_line = s->end_line;
    out.section = s->section;
    out.is_fwd_decl = s->is_fwd_decl;
    out.tag_name = s->tag_name ? strdup(s->tag_name) : NULL;
    out.n_callees = s->n_callees;
    out.cap_callees = s->n_callees;
    if (s->n_callees > 0) {
        out.callees = malloc((size_t)s->n_callees * sizeof(char *));
        for (int i = 0; i < s->n_callees; i++)
            out.callees[i] = strdup(s->callees[i]);
    }
    return out;
}

static struct file_entry deep_copy_fe(const struct file_entry *fe) {
    struct file_entry out;
    memset(&out, 0, sizeof(out));
    out.abs_path = fe->abs_path ? strdup(fe->abs_path) : NULL;
    out.rel_path = fe->rel_path ? strdup(fe->rel_path) : NULL;
    out.n_syms = fe->n_syms;
    out.cap_syms = fe->n_syms;
    if (fe->n_syms > 0) {
        out.syms = malloc((size_t)fe->n_syms * sizeof(struct symbol));
        for (int i = 0; i < fe->n_syms; i++)
            out.syms[i] = deep_copy_symbol(&fe->syms[i]);
    }
    out.n_includes = fe->n_includes;
    if (fe->n_includes > 0) {
        out.includes = malloc((size_t)fe->n_includes * sizeof(char *));
        for (int i = 0; i < fe->n_includes; i++)
            out.includes[i] = strdup(fe->includes[i]);
    }
    return out;
}

static void free_cache_entry_contents(struct cache_entry *ce) {
    free(ce->abs_path);
    /* Free the deep-copied file_entry innards */
    for (int i = 0; i < ce->fe.n_syms; i++) {
        free(ce->fe.syms[i].text);
        free(ce->fe.syms[i].tag_name);
        for (int k = 0; k < ce->fe.syms[i].n_callees; k++)
            free(ce->fe.syms[i].callees[k]);
        free(ce->fe.syms[i].callees);
    }
    free(ce->fe.syms);
    free(ce->fe.abs_path);
    free(ce->fe.rel_path);
    for (int i = 0; i < ce->fe.n_includes; i++)
        free(ce->fe.includes[i]);
    free(ce->fe.includes);
    for (int i = 0; i < ce->n_tags; i++)
        free(ce->tag_names[i]);
    free(ce->tag_names);
}

/* ── Binary serialization helpers ────────────────────────────────────── */

static void write_u8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static void write_u16(FILE *f, uint16_t v) { fwrite(&v, 2, 1, f); }
static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

static void write_str(FILE *f, const char *s) {
    if (!s) { write_u16(f, 0); return; }
    uint16_t len = (uint16_t)strlen(s);
    write_u16(f, len);
    fwrite(s, 1, len, f);
}

static void write_str32(FILE *f, const char *s) {
    if (!s) { write_u32(f, 0); return; }
    uint32_t len = (uint32_t)strlen(s);
    write_u32(f, len);
    fwrite(s, 1, len, f);
}

static int read_u8(FILE *f, uint8_t *v) { return fread(v, 1, 1, f) == 1; }
static int read_u16(FILE *f, uint16_t *v) { return fread(v, 2, 1, f) == 1; }
static int read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE *f, uint64_t *v) { return fread(v, 8, 1, f) == 1; }

static char *read_str(FILE *f) {
    uint16_t len;
    if (!read_u16(f, &len)) return NULL;
    if (len == 0) return NULL;
    char *s = malloc(len + 1);
    if (fread(s, 1, len, f) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

static char *read_str32(FILE *f) {
    uint32_t len;
    if (!read_u32(f, &len)) return NULL;
    if (len == 0) return NULL;
    char *s = malloc(len + 1);
    if (fread(s, 1, len, f) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

/* ── Serialize / deserialize cache entries ────────────────────────────── */

static void serialize_entry(FILE *f, const struct cache_entry *ce) {
    fwrite(ce->blob_oid.id, 1, GIT_OID_SHA1_SIZE, f);

    /* path */
    uint16_t path_len = (uint16_t)strlen(ce->abs_path);
    write_u16(f, path_len);
    fwrite(ce->abs_path, 1, path_len, f);

    write_u16(f, (uint16_t)ce->fe.n_syms);
    write_u16(f, (uint16_t)ce->fe.n_includes);
    write_u16(f, (uint16_t)ce->n_tags);

    /* symbols */
    for (int i = 0; i < ce->fe.n_syms; i++) {
        const struct symbol *s = &ce->fe.syms[i];
        write_str32(f, s->text);
        write_u16(f, (uint16_t)s->n_callees);
        write_u16(f, (uint16_t)s->cap_idx);
        write_u32(f, s->start_line);
        write_u32(f, s->end_line);
        write_u8(f, (uint8_t)(s->section & 0xff));
        write_u8(f, (uint8_t)s->is_fwd_decl);
        write_str(f, s->tag_name);
        for (int k = 0; k < s->n_callees; k++)
            write_str(f, s->callees[k]);
    }

    /* includes */
    for (int i = 0; i < ce->fe.n_includes; i++)
        write_str(f, ce->fe.includes[i]);

    /* tag names */
    for (int i = 0; i < ce->n_tags; i++)
        write_str(f, ce->tag_names[i]);
}

static int deserialize_entry(FILE *f, struct cache_entry *ce) {
    memset(ce, 0, sizeof(*ce));

    if (fread(ce->blob_oid.id, 1, GIT_OID_SHA1_SIZE, f) != GIT_OID_SHA1_SIZE)
        return -1;

    uint16_t path_len;
    if (!read_u16(f, &path_len)) return -1;
    ce->abs_path = malloc(path_len + 1);
    if (fread(ce->abs_path, 1, path_len, f) != path_len) {
        free(ce->abs_path);
        return -1;
    }
    ce->abs_path[path_len] = '\0';

    uint16_t n_syms, n_includes, n_tags;
    if (!read_u16(f, &n_syms) || !read_u16(f, &n_includes) ||
        !read_u16(f, &n_tags))
        return -1;

    ce->fe.n_syms = n_syms;
    ce->fe.cap_syms = n_syms;
    ce->fe.abs_path = strdup(ce->abs_path);
    if (n_syms > 0) {
        ce->fe.syms = malloc((size_t)n_syms * sizeof(struct symbol));
        for (int i = 0; i < n_syms; i++) {
            struct symbol *s = &ce->fe.syms[i];
            memset(s, 0, sizeof(*s));
            s->text = read_str32(f);
            uint16_t nc, ci;
            if (!read_u16(f, &nc) || !read_u16(f, &ci)) return -1;
            s->n_callees = nc;
            s->cap_callees = nc;
            s->cap_idx = ci;
            if (!read_u32(f, &s->start_line) || !read_u32(f, &s->end_line))
                return -1;
            uint8_t sec, fwd;
            if (!read_u8(f, &sec) || !read_u8(f, &fwd)) return -1;
            s->section = (int)(int8_t)sec;
            s->is_fwd_decl = fwd;
            s->tag_name = read_str(f);
            if (nc > 0) {
                s->callees = malloc((size_t)nc * sizeof(char *));
                for (int k = 0; k < nc; k++)
                    s->callees[k] = read_str(f);
            }
        }
    }

    ce->fe.n_includes = n_includes;
    if (n_includes > 0) {
        ce->fe.includes = malloc((size_t)n_includes * sizeof(char *));
        for (int i = 0; i < n_includes; i++)
            ce->fe.includes[i] = read_str(f);
    }

    ce->n_tags = n_tags;
    if (n_tags > 0) {
        ce->tag_names = malloc((size_t)n_tags * sizeof(char *));
        for (int i = 0; i < n_tags; i++)
            ce->tag_names[i] = read_str(f);
    }

    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────── */

char *cache_make_opts_str(int show_calls, char **defines, int n_defines,
                          char **strip_macros, int n_strip,
                          char **inc_paths, int n_inc) {
    char *buf = NULL;
    size_t buf_len = 0;
    FILE *m = open_memstream(&buf, &buf_len);
    if (!m) return strdup("");

    fprintf(m, "v=%d\ncalls=%d\n", CACHE_VERSION, show_calls);

    fprintf(m, "defines=");
    for (int i = 0; i < n_defines; i++)
        fprintf(m, "%s%s", defines[i], i < n_defines - 1 ? "," : "");
    fprintf(m, "\nstrips=");
    for (int i = 0; i < n_strip; i++)
        fprintf(m, "%s%s", strip_macros[i], i < n_strip - 1 ? "," : "");
    fprintf(m, "\ninc_paths=");
    for (int i = 0; i < n_inc; i++)
        fprintf(m, "%s%s", inc_paths[i], i < n_inc - 1 ? "," : "");
    fprintf(m, "\n");
    fclose(m);
    return buf;
}

archmap_cache *cache_open(const char *work_dir, const char *opts_str) {
    git_libgit2_init();

    git_repository *repo = NULL;
    int err = git_repository_open_ext(&repo, work_dir, 0, NULL);
    if (err != 0 || !repo) {
        git_libgit2_shutdown();
        return NULL;
    }

    archmap_cache *c = calloc(1, sizeof(*c));
    c->repo = repo;
    c->opts_hash = fnv1a(opts_str);

    /* Build cache path: <git_dir>/archmap-cache */
    const char *git_dir = git_repository_path(repo);
    snprintf(c->cache_path, sizeof(c->cache_path),
             "%sarchmap-cache", git_dir);

    /* Try to read existing cache */
    FILE *f = fopen(c->cache_path, "rb");
    if (f) {
        char magic[4];
        uint32_t version, n_entries;
        uint64_t opts_hash;
        int valid = 1;

        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CACHE_MAGIC, 4) != 0)
            valid = 0;
        if (valid && (!read_u32(f, &version) || version != CACHE_VERSION))
            valid = 0;
        if (valid && (!read_u64(f, &opts_hash) || opts_hash != c->opts_hash))
            valid = 0;
        if (valid && !read_u32(f, &n_entries))
            valid = 0;

        if (valid) {
            /* Skip 12 bytes reserved */
            fseek(f, 12, SEEK_CUR);

            c->cap_entries = (int)n_entries > 0 ? (int)n_entries : 16;
            c->entries = malloc((size_t)c->cap_entries * sizeof(struct cache_entry));

            for (uint32_t i = 0; i < n_entries; i++) {
                if (deserialize_entry(f, &c->entries[c->n_entries]) == 0)
                    c->n_entries++;
                else
                    break;
            }
        }
        fclose(f);
    }

    if (!c->entries) {
        c->cap_entries = 64;
        c->entries = malloc((size_t)c->cap_entries * sizeof(struct cache_entry));
    }

    c->cap_fresh = 64;
    c->fresh = malloc((size_t)c->cap_fresh * sizeof(struct cache_entry));

    return c;
}

int cache_lookup(archmap_cache *c, const char *abs_path,
                 struct file_entry *out_fe,
                 char ***out_tags, int *out_n_tags) {
    if (!c) return 0;

    git_oid oid;
    if (git_repository_hashfile(&oid, c->repo, abs_path,
                                 GIT_OBJECT_BLOB, NULL) != 0)
        return 0;

    /* Linear scan for matching OID */
    for (int i = 0; i < c->n_entries; i++) {
        if (git_oid_equal(&oid, &c->entries[i].blob_oid)) {
            /* Deep-copy file_entry */
            *out_fe = deep_copy_fe(&c->entries[i].fe);

            /* Deep-copy tags */
            *out_n_tags = c->entries[i].n_tags;
            if (c->entries[i].n_tags > 0) {
                *out_tags = malloc((size_t)c->entries[i].n_tags * sizeof(char *));
                for (int t = 0; t < c->entries[i].n_tags; t++)
                    (*out_tags)[t] = strdup(c->entries[i].tag_names[t]);
            } else {
                *out_tags = NULL;
            }
            return 1;
        }
    }

    /* Also check fresh entries (for multi-file runs where same content appears) */
    for (int i = 0; i < c->n_fresh; i++) {
        if (git_oid_equal(&oid, &c->fresh[i].blob_oid)) {
            *out_fe = deep_copy_fe(&c->fresh[i].fe);
            *out_n_tags = c->fresh[i].n_tags;
            if (c->fresh[i].n_tags > 0) {
                *out_tags = malloc((size_t)c->fresh[i].n_tags * sizeof(char *));
                for (int t = 0; t < c->fresh[i].n_tags; t++)
                    (*out_tags)[t] = strdup(c->fresh[i].tag_names[t]);
            } else {
                *out_tags = NULL;
            }
            return 1;
        }
    }

    return 0;
}

void cache_store(archmap_cache *c, const char *abs_path,
                 const struct file_entry *fe,
                 char **tag_names, int n_tags) {
    if (!c) return;

    git_oid oid;
    if (git_repository_hashfile(&oid, c->repo, abs_path,
                                 GIT_OBJECT_BLOB, NULL) != 0)
        return;

    if (c->n_fresh >= c->cap_fresh) {
        c->cap_fresh *= 2;
        c->fresh = realloc(c->fresh,
                           (size_t)c->cap_fresh * sizeof(struct cache_entry));
    }

    struct cache_entry *ce = &c->fresh[c->n_fresh++];
    memset(ce, 0, sizeof(*ce));
    git_oid_cpy(&ce->blob_oid, &oid);
    ce->abs_path = strdup(abs_path);
    ce->fe = deep_copy_fe(fe);
    ce->n_tags = n_tags;
    if (n_tags > 0) {
        ce->tag_names = malloc((size_t)n_tags * sizeof(char *));
        for (int i = 0; i < n_tags; i++)
            ce->tag_names[i] = strdup(tag_names[i]);
    }

    c->dirty = 1;
}

void cache_close(archmap_cache *c) {
    if (!c) return;

    if (c->dirty || c->n_fresh > 0) {
        /* Merge: fresh entries win on OID collision with old entries.
           Only keep entries that appeared in this run (fresh). */
        int total = c->n_fresh;
        struct cache_entry *merged = malloc(
            (size_t)(total > 0 ? total : 1) * sizeof(struct cache_entry));

        for (int i = 0; i < c->n_fresh; i++)
            merged[i] = c->fresh[i];  /* shallow move */

        /* Write atomically: tmp + rename */
        char tmp_path[PATH_MAX];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", c->cache_path);
        FILE *f = fopen(tmp_path, "wb");
        if (f) {
            /* Header: 32 bytes */
            fwrite(CACHE_MAGIC, 1, 4, f);
            write_u32(f, CACHE_VERSION);
            write_u64(f, c->opts_hash);
            write_u32(f, (uint32_t)total);
            /* 12 bytes reserved */
            uint8_t reserved[12] = {0};
            fwrite(reserved, 1, 12, f);

            for (int i = 0; i < total; i++)
                serialize_entry(f, &merged[i]);

            fclose(f);
            rename(tmp_path, c->cache_path);
        }

        /* Free fresh entries (they were shallow-moved into merged) */
        for (int i = 0; i < total; i++)
            free_cache_entry_contents(&merged[i]);
        free(merged);

        /* fresh array was shallow-moved, just free the container */
        free(c->fresh);
        c->fresh = NULL;
        c->n_fresh = 0;
    } else {
        for (int i = 0; i < c->n_fresh; i++)
            free_cache_entry_contents(&c->fresh[i]);
        free(c->fresh);
    }

    /* Free old entries */
    for (int i = 0; i < c->n_entries; i++)
        free_cache_entry_contents(&c->entries[i]);
    free(c->entries);

    git_repository_free(c->repo);
    git_libgit2_shutdown();
    free(c);
}
