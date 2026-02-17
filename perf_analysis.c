/* perf_analysis.c — Performance analysis pipeline for archmap
 *
 * Pipeline: build → profile (direct perf_event_open + libdwfl + capstone) →
 *           skeleton cross-reference → compact report
 *
 * Single execution per profile: counters, sampling, and topdown all run
 * simultaneously on the same fork+exec child via perf_event_open().
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <fts.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/personality.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <math.h>

#include <linux/perf_event.h>
#include <perfmon/pfmlib_perf_event.h>
#include <elfutils/libdwfl.h>
#include <libelf.h>
#include <gelf.h>
#include <capstone/capstone.h>

#include <tree_sitter/api.h>
#include "perf_analysis.h"
#include "git_cache.h"

#ifndef PERF_FLAG_FD_CLOEXEC
#define PERF_FLAG_FD_CLOEXEC (1UL << 3)
#endif

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

static void arena_init(struct arena *a, size_t block_size) {
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

static void *arena_alloc(struct arena *a, size_t n) {
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

static char *arena_strdup(struct arena *a, const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *p = arena_alloc(a, len);
    memcpy(p, s, len);
    return p;
}

static void *arena_realloc(struct arena *a, void *old_ptr,
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

static size_t arena_save(struct arena *a) {
    return a->head ? a->head->used : 0;
}

static void arena_reset(struct arena *a, size_t mark) {
    if (a->head) a->head->used = mark;
}

static void arena_destroy(struct arena *a) {
    struct arena_block *b = a->head;
    while (b) {
        struct arena_block *next = b->next;
        munmap(b, b->size);
        b = next;
    }
    a->head = NULL;
}

/* ── String intern table ─────────────────────────────────────────────── */

/* wyhash core (copied from hmap_avx512.c) */

static inline uint64_t _pa_wymix(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

static inline uint64_t _pa_wyr8(const void *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

static inline uint64_t _pa_wyr4(const void *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

static inline uint64_t _pa_wyr3(const void *p, size_t k) {
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
            a = (_pa_wyr4(p) << 32) | _pa_wyr4(p + ((len >> 3) << 2));
            b = (_pa_wyr4(p + len - 4) << 32) |
                _pa_wyr4(p + len - 4 - ((len >> 3) << 2));
        } else if (__builtin_expect(len > 0, 1)) {
            a = _pa_wyr3(p, len);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                seed = _pa_wymix(_pa_wyr8(p)      ^ s1,
                                 _pa_wyr8(p + 8)  ^ seed);
                see1 = _pa_wymix(_pa_wyr8(p + 16) ^ s2,
                                 _pa_wyr8(p + 24) ^ see1);
                see2 = _pa_wymix(_pa_wyr8(p + 32) ^ s3,
                                 _pa_wyr8(p + 40) ^ see2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = _pa_wymix(_pa_wyr8(p) ^ s1, _pa_wyr8(p + 8) ^ seed);
            i -= 16; p += 16;
        }
        a = _pa_wyr8(p + i - 16);
        b = _pa_wyr8(p + i - 8);
    }
    return _pa_wymix(s1 ^ len, _pa_wymix(a ^ s1, b ^ seed));
}

struct intern_slot {
    uint64_t hash;
    const char *str;
};

struct intern_table {
    struct intern_slot *slots;
    uint32_t cap, count;
    struct arena arena;
};

static void intern_init(struct intern_table *t, uint32_t initial_cap) {
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

static const char *intern_str(struct intern_table *t, const char *s) {
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

static void intern_destroy(struct intern_table *t) {
    free(t->slots);
    t->slots = NULL;
    arena_destroy(&t->arena);
    t->count = t->cap = 0;
}

/* ── Profile data ────────────────────────────────────────────────────── */

struct perf_profile {
    struct perf_stats stats;
    struct hot_func *funcs;
    struct hot_insn *insns;
    struct uprof_func *uprof_funcs;
    struct mca_block *mca_blocks;
    struct cache_miss_site *cm_sites;
    struct mem_hotspot *mem_hotspots;
    struct struct_layout *layouts;
    struct remark_entry *remarks;
    int n_funcs, n_insns;
    int n_uprof_funcs, n_mca_blocks;
    int n_cm_sites, n_mem_hotspots;
    int n_layouts, n_remarks;
    int has_topdown, n_runs;
    struct topdown_metrics topdown;
    struct run_stats rs_cycles, rs_insns, rs_ipc, rs_wall;
    struct run_stats rs_cache_miss_pct, rs_branch_miss_pct;
    struct arena arena;
    struct intern_table strings;
};

/* ── Options ─────────────────────────────────────────────────────────── */

struct perf_opts {
    int top_n;
    int insns_n;
    int n_runs;           /* 0=auto (1 for single, 5 for A/B), >0=explicit */
    int cmd_argc;
    unsigned no_build : 1;
    unsigned verbose  : 1;
    int uprof_mode  : 2;  /* 1=force, -1=skip, 0=auto */
    int topdown_mode: 2;
    int mca_mode    : 2;
    int cachemiss_mode: 2;
    int pahole_mode : 2;
    int remarks_mode: 2;  /* 0=auto, 1=force, -1=skip */
    const char *build_cmd;
    const char *source_dir;
    char **cmd_argv;
    char *binary_path;
    char *cmd_str;
    const char *vs_binary;
};

enum {
    PERF_OPT_NO_BUILD = 256,
    PERF_OPT_UPROF,
    PERF_OPT_NO_UPROF,
    PERF_OPT_TOPDOWN,
    PERF_OPT_NO_TOPDOWN,
    PERF_OPT_MCA,
    PERF_OPT_NO_MCA,
    PERF_OPT_CACHEMISS,
    PERF_OPT_NO_CACHEMISS,
    PERF_OPT_PAHOLE,
    PERF_OPT_NO_PAHOLE,
    PERF_OPT_VS,
    PERF_OPT_REMARKS,
    PERF_OPT_NO_REMARKS,
    PERF_OPT_RUNS,
};

static struct option perf_long_options[] = {
    {"top",       required_argument, NULL, 'n'},
    {"insns",     required_argument, NULL, 'i'},
    {"build-cmd", required_argument, NULL, 'b'},
    {"no-build",  no_argument,       NULL, PERF_OPT_NO_BUILD},
    {"uprof",          no_argument,       NULL, PERF_OPT_UPROF},
    {"no-uprof",       no_argument,       NULL, PERF_OPT_NO_UPROF},
    {"topdown",        no_argument,       NULL, PERF_OPT_TOPDOWN},
    {"no-topdown",     no_argument,       NULL, PERF_OPT_NO_TOPDOWN},
    {"mca",            no_argument,       NULL, PERF_OPT_MCA},
    {"no-mca",         no_argument,       NULL, PERF_OPT_NO_MCA},
    {"cache-misses",   no_argument,       NULL, PERF_OPT_CACHEMISS},
    {"no-cache-misses",no_argument,       NULL, PERF_OPT_NO_CACHEMISS},
    {"pahole",         no_argument,       NULL, PERF_OPT_PAHOLE},
    {"no-pahole",      no_argument,       NULL, PERF_OPT_NO_PAHOLE},
    {"vs",             required_argument, NULL, PERF_OPT_VS},
    {"remarks",        no_argument,       NULL, PERF_OPT_REMARKS},
    {"no-remarks",     no_argument,       NULL, PERF_OPT_NO_REMARKS},
    {"runs",           required_argument, NULL, PERF_OPT_RUNS},
    {"source",         required_argument, NULL, 's'},
    {"verbose",   no_argument,       NULL, 'v'},
    {"help",      no_argument,       NULL, 'h'},
    {NULL, 0, NULL, 0}
};

static void perf_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s perf [options] -- <command> [args...]\n\n"
        "Options:\n"
        "  -n, --top N          Top N hot functions (default 10)\n"
        "  -i, --insns N        Hot instructions per function (default 10)\n"
        "  -b, --build-cmd CMD  Build command (default \"make\")\n"
        "  --no-build           Skip build\n"
        "  --uprof              Force AMDuProf\n"
        "  --no-uprof           Skip AMDuProf\n"
        "  --topdown / --no-topdown     Force/skip TMA top-down analysis\n"
        "  --mca / --no-mca             Force/skip llvm-mca throughput analysis\n"
        "  --cache-misses / --no-cache-misses  Force/skip cache miss attribution\n"
        "  --pahole / --no-pahole       Force/skip struct layout analysis\n"
        "  --remarks / --no-remarks     Force/skip compiler optimization remarks\n"
        "  --vs BINARY                  A/B comparison: baseline binary (A)\n"
        "  --runs N                     Number of profiling runs (default: 1, A/B: 5)\n"
        "  -s, --source DIR     Source dir for skeleton xref (default \".\")\n"
        "  -v, --verbose        Print tool commands\n",
        prog);
}

/* ── Utility ─────────────────────────────────────────────────────────── */

static char g_tmpdir[PATH_MAX];

static void cleanup_tmpdir(void) {
    if (g_tmpdir[0] == '\0') return;
    char cmd[PATH_MAX + 16];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", g_tmpdir);
    system(cmd);
    g_tmpdir[0] = '\0';
}

/* ── OOM-aborting allocator wrappers ──────────────────────────────────── */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) { fprintf(stderr, "out of memory (malloc %zu)\n", n); abort(); }
    return p;
}
static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p && count && size) { fprintf(stderr, "out of memory (calloc %zu)\n", count * size); abort(); }
    return p;
}
static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p && n) { fprintf(stderr, "out of memory (realloc %zu)\n", n); abort(); }
    return p;
}

static char *run_cmd(const char *cmd, int *out_status, int verbose) {
    if (verbose) fprintf(stderr, "+ %s\n", cmd);
    FILE *fp = popen(cmd, "r");
    if (!fp) { if (out_status) *out_status = -1; return NULL; }

    size_t cap = 8192, len = 0;
    char *buf = xmalloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, fp)) > 0) {
        len += n;
        if (len + 1 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
    }
    buf[len] = '\0';

    int status = pclose(fp);
    if (out_status)
        *out_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return buf;
}

static int needs_shell_quote(const char *s) {
    for (; *s; s++) {
        if (*s == ' ' || *s == '\t' || *s == '&' || *s == '|' ||
            *s == ';' || *s == '(' || *s == ')' || *s == '<' ||
            *s == '>' || *s == '*' || *s == '?' || *s == '[' ||
            *s == '$' || *s == '`' || *s == '"' || *s == '\\' ||
            *s == '\n' || *s == '!')
            return 1;
    }
    return 0;
}

static char *join_argv(char **argv, int argc) {
    size_t total = 0;
    for (int i = 0; i < argc; i++)
        total += strlen(argv[i]) * 4 + 4;
    char *cmd = xmalloc(total + 1);
    char *p = cmd;
    for (int i = 0; i < argc; i++) {
        if (i > 0) *p++ = ' ';
        if (needs_shell_quote(argv[i])) {
            *p++ = '\'';
            for (const char *c = argv[i]; *c; c++) {
                if (*c == '\'') {
                    memcpy(p, "'\\''", 4);
                    p += 4;
                } else {
                    *p++ = *c;
                }
            }
            *p++ = '\'';
        } else {
            size_t al = strlen(argv[i]);
            memcpy(p, argv[i], al);
            p += al;
        }
    }
    *p = '\0';
    return cmd;
}

static void fmt_count(char *buf, size_t sz, uint64_t val) {
    if (val >= 1000000000ULL)
        snprintf(buf, sz, "%.2fG", val / 1e9);
    else if (val >= 1000000ULL)
        snprintf(buf, sz, "%.1fM", val / 1e6);
    else if (val >= 1000ULL)
        snprintf(buf, sz, "%.0fK", val / 1e3);
    else
        snprintf(buf, sz, "%" PRIu64, val);
}

static void strip_compiler_suffix(char *name) {
    char *p;
    if ((p = strstr(name, ".cold")))      *p = '\0';
    if ((p = strstr(name, ".isra.")))     *p = '\0';
    if ((p = strstr(name, ".constprop."))) *p = '\0';
    if ((p = strstr(name, ".part.")))     *p = '\0';
    if ((p = strstr(name, ".lto_priv."))) *p = '\0';
}

static char *resolve_binary(const char *name) {
    /* Absolute or relative path */
    if (name[0] == '/' || name[0] == '.') {
        char *real = realpath(name, NULL);
        if (real) return real;
    }
    /* Search PATH */
    const char *path_env = getenv("PATH");
    if (path_env) {
        char buf[PATH_MAX];
        const char *p = path_env;
        while (*p) {
            const char *end = strchr(p, ':');
            size_t dlen = end ? (size_t)(end - p) : strlen(p);
            if (dlen > 0 && dlen + 1 + strlen(name) < PATH_MAX) {
                memcpy(buf, p, dlen);
                buf[dlen] = '/';
                strcpy(buf + dlen + 1, name);
                if (access(buf, X_OK) == 0) {
                    char *real = realpath(buf, NULL);
                    return real ? real : strdup(buf);
                }
            }
            if (end) p = end + 1; else break;
        }
    }
    return strdup(name);
}

static int check_perf_access(void) {
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (!f) return 0;
    int val = 0;
    if (fscanf(f, "%d", &val) != 1) val = 0;
    fclose(f);
    if (val >= 2 && getuid() != 0) {
        fprintf(stderr,
            "error: perf_event_paranoid=%d (>=2) and not root\n"
            "  Run: sudo sysctl kernel.perf_event_paranoid=1\n", val);
        return -1;
    }
    return 0;
}

static int has_tool(const char *name) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "which '%s' >/dev/null 2>&1", name);
    return system(cmd) == 0;
}

/* Resolve a tool to its full path, checking PATH then well-known dirs.
   Returns static buffer — valid until next call. */
static const char *find_tool(const char *name) {
    static char path[PATH_MAX];

    /* Check PATH first */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "which '%s' 2>/dev/null", name);
    FILE *fp = popen(cmd, "r");
    if (fp) {
        if (fgets(path, sizeof(path), fp)) {
            char *nl = strchr(path, '\n');
            if (nl) *nl = '\0';
            if (pclose(fp) == 0 && path[0] == '/')
                return path;
        } else {
            pclose(fp);
        }
    }

    /* Well-known install locations */
    static const char *search_dirs[] = {
        "/opt/AMDuProf/bin",
        "/opt/AMDuProf_5.0/bin",
        "/opt/AMDuProf_4.2/bin",
        "/usr/local/bin",
        NULL
    };
    for (const char **d = search_dirs; *d; d++) {
        snprintf(path, sizeof(path), "%s/%s", *d, name);
        if (access(path, X_OK) == 0)
            return path;
    }
    return NULL;
}

/* Names that are too generic to be useful as "caller" */
static int is_boring_caller(const char *name) {
    if (name[0] == '0' && name[1] == 'x') return 1; /* hex address */
    static const char *boring[] = {
        "_start", "__libc_start_main", "__libc_start_call_main",
        "__GI___clone3", "start_thread", "__clone3", "_dl_start_user",
        NULL
    };
    for (const char **b = boring; *b; b++)
        if (strcmp(name, *b) == 0) return 1;
    return 0;
}

static void add_caller_entry(struct perf_profile *prof, struct hot_func *hf,
                              const char *name, double pct) {
    /* Filter: @plt stubs, __x86_ thunks */
    if (strstr(name, "@plt") || strncmp(name, "__x86_", 6) == 0) return;
    /* Filter: self-recursive (compare with stripped LTO suffixes) */
    char cn[256], cf[256];
    snprintf(cn, sizeof(cn), "%s", name);
    strip_compiler_suffix(cn);
    snprintf(cf, sizeof(cf), "%s", hf->name);
    strip_compiler_suffix(cf);
    if (strcmp(cn, cf) == 0) return;
    /* Merge duplicate caller names */
    for (int k = 0; k < hf->n_callers; k++) {
        if (strcmp(hf->callers[k].name, name) == 0) {
            hf->callers[k].pct += pct;
            return;
        }
    }
    if (hf->n_callers >= 5) return;
    if (!hf->callers)
        hf->callers = arena_alloc(&prof->arena, 5 * sizeof(*hf->callers));
    hf->callers[hf->n_callers].name =
        (char *)intern_str(&prof->strings, name);
    hf->callers[hf->n_callers].pct = pct;
    hf->n_callers++;
}

/* ── Phase 1: Build ──────────────────────────────────────────────────── */

static int phase_build(struct perf_opts *opts) {
    if (opts->no_build) return 0;
    fprintf(stderr, "building: %s\n", opts->build_cmd);
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "%s 2>&1", opts->build_cmd);
    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    if (status != 0) {
        fprintf(stderr, "build failed (exit %d):\n%s\n", status, out ? out : "");
        free(out);
        return -1;
    }
    free(out);
    return 0;
}

/* ── Raw sample structure ────────────────────────────────────────────── */

struct raw_sample {
    uint64_t ip;
    uint64_t addr;      /* data virtual address (0 if not sampled) */
    uint64_t *ips;      /* callchain IPs (NULL if no callchain) */
    int n_ips;
};

/* ── Counters — replaces run_perf_stat() ──────────────────────────────
   Each counter is opened independently (no PERF_FORMAT_GROUP) so the
   kernel can multiplex them individually alongside sampling events.
   A 6-event group would monopolise all 6 GP PMU counters on AMD Zen,
   leaving no room for sampling events and causing the instruction
   counter to return bogus values under heavy multiplexing. */

#define N_COUNTERS 6

struct counter_group { int fds[N_COUNTERS]; };

struct counter_single_read {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
};

static int counters_open(pid_t child, struct counter_group *cg) {
    static const uint64_t events[N_COUNTERS] = {
        PERF_COUNT_HW_CPU_CYCLES,
        PERF_COUNT_HW_INSTRUCTIONS,
        PERF_COUNT_HW_CACHE_REFERENCES,
        PERF_COUNT_HW_CACHE_MISSES,
        PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
        PERF_COUNT_HW_BRANCH_MISSES,
    };

    for (int i = 0; i < N_COUNTERS; i++) cg->fds[i] = -1;

    for (int i = 0; i < N_COUNTERS; i++) {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(attr);
        attr.config = events[i];
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.disabled = 1;
        attr.enable_on_exec = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                           PERF_FORMAT_TOTAL_TIME_RUNNING;

        cg->fds[i] = (int)syscall(SYS_perf_event_open, &attr, child,
                                   -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (cg->fds[i] < 0) {
            fprintf(stderr, "warning: perf_event_open counter %d: %s\n",
                    i, strerror(errno));
            /* Non-fatal: leave fd as -1, read will skip it */
        }
    }
    /* At minimum we need cycles (fd[0]) */
    return (cg->fds[0] >= 0) ? 0 : -1;
}

static uint64_t read_one_counter(int fd) {
    if (fd < 0) return 0;
    struct counter_single_read d;
    if (read(fd, &d, sizeof(d)) < (ssize_t)sizeof(d)) return 0;
    if (d.time_running == 0) return 0;
    /* If the counter ran less than 5% of the time, the scaled value is
       unreliable (small noise × large scale = garbage).  Report 0. */
    if (d.time_running * 20 < d.time_enabled) return 0;
    return (uint64_t)((double)d.value *
                      (double)d.time_enabled / (double)d.time_running);
}

static int counters_read(struct counter_group *cg, struct perf_stats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (cg->fds[0] < 0) return -1;

    stats->cycles       = read_one_counter(cg->fds[0]);
    stats->instructions = read_one_counter(cg->fds[1]);
    stats->cache_refs   = read_one_counter(cg->fds[2]);
    stats->cache_misses = read_one_counter(cg->fds[3]);
    stats->branches     = read_one_counter(cg->fds[4]);
    stats->branch_misses = read_one_counter(cg->fds[5]);

    if (stats->cycles > 0)
        stats->ipc = (double)stats->instructions / stats->cycles;
    if (stats->cache_refs > 0)
        stats->cache_miss_pct =
            100.0 * stats->cache_misses / stats->cache_refs;
    if (stats->branches > 0)
        stats->branch_miss_pct =
            100.0 * stats->branch_misses / stats->branches;

    return 0;
}

static void counters_close(struct counter_group *cg) {
    for (int i = 0; i < N_COUNTERS; i++) {
        if (cg->fds[i] >= 0) close(cg->fds[i]);
        cg->fds[i] = -1;
    }
}

/* ── Topdown — replaces run_topdown_intel/amd() ──────────────────────
   Each event opened independently (same as counters) so each gets its
   own multiplexing scale factor.  This matches what `perf stat -M`
   does and avoids group-scheduling skew where the 4-event group only
   runs during non-representative workload phases. */

#define MAX_TOPDOWN 4

enum topdown_model { TOPDOWN_INTEL, TOPDOWN_AMD };

struct topdown_group {
    int fds[MAX_TOPDOWN];
    int n_events;
    enum topdown_model model;
    int dispatch_width;     /* AMD only: 8 for Zen5+, 6 for Zen2-4 */
};

static int topdown_try_events(pid_t child, struct topdown_group *tg,
                               const char **event_names, int n_events,
                               enum topdown_model model) {
    for (int i = 0; i < MAX_TOPDOWN; i++) tg->fds[i] = -1;
    tg->n_events = 0;
    tg->model = model;

    for (int i = 0; i < n_events && i < MAX_TOPDOWN; i++) {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);

        pfm_perf_encode_arg_t arg;
        memset(&arg, 0, sizeof(arg));
        arg.attr = &attr;
        arg.size = sizeof(arg);

        int ret = pfm_get_os_event_encoding(event_names[i],
                      PFM_PLM3, PFM_OS_PERF_EVENT, &arg);
        if (ret != PFM_SUCCESS) {
            for (int j = 0; j < i; j++) { close(tg->fds[j]); tg->fds[j] = -1; }
            tg->n_events = 0;
            return -1;
        }

        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.disabled = 1;
        attr.enable_on_exec = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                           PERF_FORMAT_TOTAL_TIME_RUNNING;

        tg->fds[i] = (int)syscall(SYS_perf_event_open, &attr, child,
                                   -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (tg->fds[i] < 0) {
            for (int j = 0; j < i; j++) { close(tg->fds[j]); tg->fds[j] = -1; }
            tg->n_events = 0;
            return -1;
        }
        tg->n_events++;
    }
    return 0;
}

/* Detect AMD dispatch width from CPUID family:
   Zen5 (family 0x1A+) = 8 wide, Zen2/3/4 (family 0x17-0x19) = 6 wide */
static int amd_dispatch_width(void) {
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return 6;
    char line[256];
    int family = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "cpu family : %d", &family) == 1) break;
    }
    fclose(fp);
    return (family >= 0x1A) ? 8 : 6;
}

static int topdown_open(pid_t child, struct topdown_group *tg, int mode) {
    memset(tg, 0, sizeof(*tg));
    for (int i = 0; i < MAX_TOPDOWN; i++) tg->fds[i] = -1;
    if (mode < 0) return -1;

    pfm_initialize();

    /* Try Intel topdown events */
    static const char *intel_events[] = {
        "TOPDOWN_RETIRING:u",
        "TOPDOWN_BAD_SPEC:u",
        "TOPDOWN_FE_BOUND:u",
        "TOPDOWN_BE_BOUND:u",
    };
    if (topdown_try_events(child, tg, intel_events, 4, TOPDOWN_INTEL) == 0)
        return 0;

    /* AMD Zen: retired_ops, backend_stalls, frontend_stalls, cycles
       Topdown L1 = each / (dispatch_width × cycles) */
    static const char *amd_events[] = {
        "RETIRED_OPS:u",
        "DISPATCH_STALLS_1:BE_STALLS:u",
        "DISPATCH_STALLS_1:FE_NO_OPS:u",
        "CYCLES_NOT_IN_HALT:u",
    };
    if (topdown_try_events(child, tg, amd_events, 4, TOPDOWN_AMD) == 0) {
        tg->dispatch_width = amd_dispatch_width();
        return 0;
    }

    if (mode > 0)
        fprintf(stderr,
            "error: --topdown specified but no topdown events "
            "available (need Intel Icelake+ or AMD Zen)\n");
    return -1;
}

static int topdown_read(struct topdown_group *tg, struct topdown_metrics *td) {
    memset(td, 0, sizeof(*td));
    if (tg->fds[0] < 0 || tg->n_events == 0) return -1;

    /* Read each event individually with its own multiplexing scale */
    uint64_t vals[MAX_TOPDOWN];
    for (int i = 0; i < tg->n_events; i++)
        vals[i] = read_one_counter(tg->fds[i]);

    td->level = 1;

    if (tg->model == TOPDOWN_INTEL) {
        uint64_t total = 0;
        for (int i = 0; i < tg->n_events; i++)
            total += vals[i];
        if (total == 0) return -1;
        td->retiring = 100.0 * vals[0] / total;
        td->bad_spec = 100.0 * vals[1] / total;
        td->frontend = 100.0 * vals[2] / total;
        td->backend  = 100.0 * vals[3] / total;
    } else {
        /* AMD: total_slots = dispatch_width × unhalted_cycles
           [0]=retired_ops  [1]=be_stalls  [2]=fe_stalls  [3]=cycles */
        uint64_t retired = vals[0];
        uint64_t be      = vals[1];
        uint64_t fe      = vals[2];
        uint64_t cycles  = vals[3];
        uint64_t total   = (uint64_t)tg->dispatch_width * cycles;
        if (total == 0) return -1;
        td->retiring = 100.0 * retired / total;
        td->backend  = 100.0 * be / total;
        td->frontend = 100.0 * fe / total;
        td->bad_spec = 100.0 - td->retiring - td->backend - td->frontend;
        if (td->bad_spec < 0) td->bad_spec = 0;
    }
    return 0;
}

static void topdown_close(struct topdown_group *tg) {
    for (int i = 0; i < tg->n_events; i++) {
        if (tg->fds[i] >= 0) close(tg->fds[i]);
        tg->fds[i] = -1;
    }
    tg->n_events = 0;
}

/* ── Sampling ring buffers — replaces run_perf_record + cache-miss ───── */

#define RING_PAGES 512  /* (1 + 512) pages ≈ 2 MB per ring */

struct sample_ring {
    int fd;
    void *mmap_base;
    size_t mmap_size;
    uint64_t sample_type;   /* stored for type-driven drain parsing */
};

struct sampling_ctx {
    struct sample_ring cycles;
    struct sample_ring cache_misses;
};

static int ring_open(pid_t child, struct sample_ring *ring,
                     uint64_t event_config, uint64_t sample_period,
                     uint64_t sample_type) {
    ring->fd = -1;
    ring->mmap_base = MAP_FAILED;
    ring->mmap_size = 0;
    ring->sample_type = sample_type;

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = event_config;
    attr.sample_period = sample_period;
    attr.sample_type = sample_type;
    attr.disabled = 1;
    attr.enable_on_exec = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    if (sample_type & PERF_SAMPLE_CALLCHAIN)
        attr.exclude_callchain_kernel = 1;

    ring->fd = (int)syscall(SYS_perf_event_open, &attr, child,
                            -1, -1, PERF_FLAG_FD_CLOEXEC);
    if (ring->fd < 0) return -1;

    long page_size = sysconf(_SC_PAGESIZE);
    ring->mmap_size = (size_t)(1 + RING_PAGES) * (size_t)page_size;
    ring->mmap_base = mmap(NULL, ring->mmap_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, ring->fd, 0);
    if (ring->mmap_base == MAP_FAILED) {
        close(ring->fd);
        ring->fd = -1;
        return -1;
    }
    return 0;
}

static int sampling_open(pid_t child, struct sampling_ctx *ctx,
                          int want_cache_misses) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->cycles.fd = -1;
    ctx->cycles.mmap_base = MAP_FAILED;
    ctx->cache_misses.fd = -1;
    ctx->cache_misses.mmap_base = MAP_FAILED;

    int ret = ring_open(child, &ctx->cycles,
                        PERF_COUNT_HW_CPU_CYCLES, 100003,
                        PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN);
    if (ret < 0) return -1;

    if (want_cache_misses) {
        /* Try with PERF_SAMPLE_ADDR for memory attribution, fall back to IP-only */
        int cm_ret = ring_open(child, &ctx->cache_misses,
                               PERF_COUNT_HW_CACHE_MISSES, 1000,
                               PERF_SAMPLE_IP | PERF_SAMPLE_ADDR);
        if (cm_ret < 0)
            ring_open(child, &ctx->cache_misses,
                      PERF_COUNT_HW_CACHE_MISSES, 1000,
                      PERF_SAMPLE_IP);
    }
    /* cache-miss ring failure is non-fatal */
    return 0;
}

static int sampling_drain(struct sample_ring *ring,
                           struct raw_sample **out, int *n_out,
                           struct arena *sa) {
    *out = NULL;
    *n_out = 0;
    if (ring->fd < 0 || ring->mmap_base == MAP_FAILED) return -1;

    struct perf_event_mmap_page *header =
        (struct perf_event_mmap_page *)ring->mmap_base;
    uint64_t data_head = __atomic_load_n(&header->data_head,
                                          __ATOMIC_ACQUIRE);
    uint64_t data_tail = header->data_tail;
    char *data = (char *)ring->mmap_base + (size_t)header->data_offset;
    uint64_t data_size = header->data_size;

    int cap = 4096;
    *out = arena_alloc(sa, (size_t)cap * sizeof(struct raw_sample));

    while (data_tail < data_head) {
        /* Read header, handling wrap-around */
        struct perf_event_header hdr;
        uint64_t off = data_tail % data_size;
        if (off + sizeof(hdr) > data_size) {
            size_t first = (size_t)(data_size - off);
            memcpy(&hdr, data + off, first);
            memcpy((char *)&hdr + first, data, sizeof(hdr) - first);
        } else {
            memcpy(&hdr, data + off, sizeof(hdr));
        }

        if (hdr.size == 0) break; /* corrupted */

        if (hdr.type == PERF_RECORD_SAMPLE) {
            /* Copy payload handling wrap */
            size_t payload = hdr.size - sizeof(hdr);
            if (payload > 0 && payload < 8192) {
                char record[8192];
                uint64_t rec_off = (data_tail + sizeof(hdr)) % data_size;
                if (rec_off + payload > data_size) {
                    size_t first = (size_t)(data_size - rec_off);
                    memcpy(record, data + rec_off, first);
                    memcpy(record + first, data, payload - first);
                } else {
                    memcpy(record, data + rec_off, payload);
                }

                if (*n_out >= cap) {
                    int new_cap = cap * 2;
                    struct raw_sample *nb = arena_alloc(
                        sa, (size_t)new_cap * sizeof(struct raw_sample));
                    memcpy(nb, *out, (size_t)cap * sizeof(struct raw_sample));
                    *out = nb;
                    cap = new_cap;
                }
                struct raw_sample *s = &(*out)[(*n_out)++];
                memset(s, 0, sizeof(*s));

                /* Type-driven field parsing: fields appear in bit order */
                size_t pos = 0;
                if (ring->sample_type & PERF_SAMPLE_IP) {
                    if (pos + 8 <= payload) {
                        memcpy(&s->ip, record + pos, 8);
                        pos += 8;
                    }
                }
                if (ring->sample_type & PERF_SAMPLE_ADDR) {
                    if (pos + 8 <= payload) {
                        memcpy(&s->addr, record + pos, 8);
                        pos += 8;
                    }
                }
                if (ring->sample_type & PERF_SAMPLE_CALLCHAIN) {
                    uint64_t nr;
                    if (pos + 8 <= payload) {
                        memcpy(&nr, record + pos, 8);
                        pos += 8;
                        if (nr > 0 && nr < 256 &&
                            pos + nr * 8 <= payload) {
                            s->n_ips = (int)nr;
                            s->ips = arena_alloc(sa, nr * 8);
                            memcpy(s->ips, record + pos, nr * 8);
                            pos += nr * 8;
                        }
                    }
                }
            }
        }

        data_tail += hdr.size;
    }

    __atomic_store_n(&header->data_tail, data_head, __ATOMIC_RELEASE);
    return 0;
}

static void ring_close(struct sample_ring *ring) {
    if (ring->mmap_base != MAP_FAILED && ring->mmap_size > 0) {
        munmap(ring->mmap_base, ring->mmap_size);
        ring->mmap_base = MAP_FAILED;
    }
    if (ring->fd >= 0) {
        close(ring->fd);
        ring->fd = -1;
    }
}

static void sampling_close(struct sampling_ctx *ctx) {
    ring_close(&ctx->cycles);
    ring_close(&ctx->cache_misses);
}

/* ── Symbol resolver — replaces perf report/annotate lookups ─────────── */

struct sym_resolver {
    Dwfl *dwfl;
    Dwfl_Module *mod;
    csh cs_handle;
    int cs_ok;
    Elf *elf;
    int elf_fd;
    uint64_t addr_bias;  /* subtract from runtime IP → dwfl address */
};

static const Dwfl_Callbacks offline_callbacks = {
    .find_elf = dwfl_build_id_find_elf,
    .find_debuginfo = dwfl_standard_find_debuginfo,
    .section_address = dwfl_offline_section_address,
};

/* Read /proc/pid/maps to find the load base of the binary.
   Returns 0 on failure (child already exited or binary not found). */
static uint64_t read_load_base(pid_t pid, const char *binary_path) {
    char maps_path[64];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
    FILE *fp = fopen(maps_path, "r");
    if (!fp) return 0;

    char *real_bin = realpath(binary_path, NULL);

    uint64_t load_base = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        uint64_t start;
        uint64_t offset;
        char pathname[512] = {0};

        if (sscanf(line, "%" SCNx64 "-%*x %*s %" SCNx64 " %*s %*s %511[^\n]",
                   &start, &offset, pathname) >= 2) {
            char *p = pathname;
            while (*p == ' ') p++;
            /* Strip " (deleted)" suffix */
            char *del = strstr(p, " (deleted)");
            if (del) *del = '\0';

            if (*p && ((real_bin && strcmp(p, real_bin) == 0) ||
                       strcmp(p, binary_path) == 0)) {
                load_base = start - offset;
                break;
            }
        }
    }
    fclose(fp);
    free(real_bin);
    return load_base;
}

static int symres_init(struct sym_resolver *sr, const char *binary_path,
                        uint64_t load_base) {
    memset(sr, 0, sizeof(*sr));
    sr->elf_fd = -1;

    elf_version(EV_CURRENT);

    sr->dwfl = dwfl_begin(&offline_callbacks);
    if (!sr->dwfl) return -1;

    sr->mod = dwfl_report_offline(sr->dwfl, "", binary_path, -1);
    dwfl_report_end(sr->dwfl, NULL, NULL);
    if (!sr->mod) {
        dwfl_end(sr->dwfl);
        sr->dwfl = NULL;
        return -1;
    }

    /* Open ELF directly for raw section data */
    sr->elf_fd = open(binary_path, O_RDONLY);
    if (sr->elf_fd >= 0)
        sr->elf = elf_begin(sr->elf_fd, ELF_C_READ, NULL);

    /* Compute address bias for PIE binaries */
    if (sr->elf && load_base > 0) {
        GElf_Ehdr ehdr;
        if (gelf_getehdr(sr->elf, &ehdr) && ehdr.e_type == ET_DYN) {
            Dwarf_Addr mod_low;
            dwfl_module_info(sr->mod, NULL, &mod_low, NULL,
                             NULL, NULL, NULL, NULL);
            sr->addr_bias = load_base - mod_low;
        }
    }

    /* Init capstone */
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &sr->cs_handle) == CS_ERR_OK) {
        cs_option(sr->cs_handle, CS_OPT_SYNTAX, CS_OPT_SYNTAX_ATT);
        sr->cs_ok = 1;
    }

    return 0;
}

static const char *symres_func_name(struct sym_resolver *sr, uint64_t addr) {
    if (!sr->dwfl) return NULL;
    uint64_t dwfl_addr = addr - sr->addr_bias;
    Dwfl_Module *mod = dwfl_addrmodule(sr->dwfl, dwfl_addr);
    if (!mod) return NULL;
    GElf_Off offset;
    GElf_Sym sym;
    return dwfl_module_addrinfo(mod, dwfl_addr, &offset, &sym,
                                NULL, NULL, NULL);
}

static int symres_srcline(struct sym_resolver *sr, uint64_t addr,
                           const char **file, int *line) {
    if (!sr->dwfl) return -1;
    uint64_t dwfl_addr = addr - sr->addr_bias;
    Dwfl_Module *mod = dwfl_addrmodule(sr->dwfl, dwfl_addr);
    if (!mod) return -1;
    Dwfl_Line *ln = dwfl_module_getsrc(mod, dwfl_addr);
    if (!ln) return -1;
    *file = dwfl_lineinfo(ln, NULL, line, NULL, NULL, NULL);
    return *file ? 0 : -1;
}

/* Returns runtime start address, raw bytes (caller must free), and length.
   Uses the raw symbol name (including .constprop.N etc). */
static int symres_func_range(struct sym_resolver *sr, const char *name,
                              uint64_t *start, uint8_t **bytes, size_t *len,
                              struct arena *a) {
    if (!sr->mod || !sr->elf) return -1;

    /* Find symbol via dwfl (for the runtime-adjusted address) */
    int nsyms = dwfl_module_getsymtab(sr->mod);
    GElf_Sym best_sym;
    GElf_Addr best_addr = 0;
    int found = 0;

    for (int i = 0; i < nsyms; i++) {
        GElf_Sym sym;
        GElf_Addr addr;
        const char *sname = dwfl_module_getsym_info(
            sr->mod, i, &sym, &addr, NULL, NULL, NULL);
        if (!sname || GELF_ST_TYPE(sym.st_info) != STT_FUNC) continue;
        if (strcmp(sname, name) == 0 && sym.st_size > 0) {
            best_sym = sym;
            best_addr = addr;
            found = 1;
            break;
        }
    }
    if (!found) return -1;

    *start = best_addr + sr->addr_bias;  /* convert to runtime address */
    *len = best_sym.st_size;

    /* Extract raw bytes from ELF using the original (non-biased) address */
    Elf_Scn *scn = NULL;
    while ((scn = elf_nextscn(sr->elf, scn)) != NULL) {
        GElf_Shdr shdr;
        gelf_getshdr(scn, &shdr);
        if (shdr.sh_type != SHT_PROGBITS) continue;
        if (!(shdr.sh_flags & SHF_EXECINSTR)) continue;

        if (best_sym.st_value >= shdr.sh_addr &&
            best_sym.st_value + best_sym.st_size <=
                shdr.sh_addr + shdr.sh_size) {
            Elf_Data *d = elf_getdata(scn, NULL);
            if (!d) return -1;
            uint64_t offset = best_sym.st_value - shdr.sh_addr;
            if (offset + best_sym.st_size > d->d_size) return -1;
            *bytes = arena_alloc(a, best_sym.st_size);
            memcpy(*bytes, (uint8_t *)d->d_buf + offset, best_sym.st_size);
            return 0;
        }
    }
    return -1;
}

static int symres_has_debug(struct sym_resolver *sr) {
    if (!sr->dwfl || !sr->mod) return 0;
    Dwarf_Addr bias;
    Dwarf *dbg = dwfl_module_getdwarf(sr->mod, &bias);
    return dbg != NULL;
}

static void symres_free(struct sym_resolver *sr) {
    if (sr->cs_ok) cs_close(&sr->cs_handle);
    if (sr->elf) elf_end(sr->elf);
    if (sr->elf_fd >= 0) close(sr->elf_fd);
    if (sr->dwfl) dwfl_end(sr->dwfl);
    memset(sr, 0, sizeof(*sr));
    sr->elf_fd = -1;
}

/* ── profile_child — fork+exec with instrumentation ──────────────────── */

static int profile_child(struct perf_opts *opts,
                          struct perf_stats *stats,
                          struct topdown_metrics *topdown, int *has_topdown,
                          struct raw_sample **cycle_samples, int *n_cycle,
                          struct raw_sample **cm_samples, int *n_cm,
                          uint64_t *out_load_base,
                          struct arena *sa) {
    *has_topdown = 0;
    if (cycle_samples) { *cycle_samples = NULL; *n_cycle = 0; }
    if (cm_samples) { *cm_samples = NULL; *n_cm = 0; }
    *out_load_base = 0;

    /* Pipe for synchronization: child blocks until parent is ready */
    int go_pipe[2];
    if (pipe(go_pipe) < 0) { perror("pipe"); return -1; }

    /* CLOEXEC pipe to detect when child has exec'd */
    int exec_pipe[2];
    if (pipe(exec_pipe) < 0) {
        close(go_pipe[0]); close(go_pipe[1]);
        perror("pipe");
        return -1;
    }
    fcntl(exec_pipe[0], F_SETFD, FD_CLOEXEC);
    fcntl(exec_pipe[1], F_SETFD, FD_CLOEXEC);

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        close(go_pipe[0]); close(go_pipe[1]);
        close(exec_pipe[0]); close(exec_pipe[1]);
        return -1;
    }

    if (child == 0) {
        /* ── Child ── */
        close(go_pipe[1]);
        close(exec_pipe[0]);

        /* Disable ASLR for deterministic addresses */
        personality(ADDR_NO_RANDOMIZE);

        /* Block until parent signals */
        char dummy;
        if (read(go_pipe[0], &dummy, 1) < 0) _exit(126);
        close(go_pipe[0]);

        execvp(opts->cmd_argv[0], opts->cmd_argv);
        /* If exec fails, write error byte (exec_pipe[1] still open) */
        (void)!write(exec_pipe[1], "E", 1);
        _exit(127);
    }

    /* ── Parent ── */
    close(go_pipe[0]);
    close(exec_pipe[1]);

    /* Open all perf events targeting child, with enable_on_exec */
    struct counter_group cg;
    int cg_ok = counters_open(child, &cg);

    struct topdown_group tg;
    int td_ok = topdown_open(child, &tg, opts->topdown_mode);

    struct sampling_ctx sctx;
    int samp_ok = -1;
    if (cycle_samples) {
        samp_ok = sampling_open(child, &sctx,
                                cm_samples != NULL);
    } else {
        /* No sampling requested — init with safe sentinel values */
        memset(&sctx, 0, sizeof(sctx));
        sctx.cycles.fd = -1;
        sctx.cycles.mmap_base = MAP_FAILED;
        sctx.cache_misses.fd = -1;
        sctx.cache_misses.mmap_base = MAP_FAILED;
    }

    /* Unblock child → exec triggers enable_on_exec atomically */
    if (write(go_pipe[1], "x", 1) < 0) { /* ignore */ }
    close(go_pipe[1]);

    /* Wait for exec to happen (exec_pipe[1] closed by CLOEXEC) */
    char exec_status;
    ssize_t exec_n = read(exec_pipe[0], &exec_status, 1);
    close(exec_pipe[0]);

    if (exec_n > 0) {
        /* exec failed */
        fprintf(stderr, "error: exec failed for %s\n", opts->cmd_argv[0]);
        waitpid(child, NULL, 0);
        counters_close(&cg);
        topdown_close(&tg);
        sampling_close(&sctx);
        return -1;
    }

    /* Child has exec'd — read load base from /proc/pid/maps */
    *out_load_base = read_load_base(child, opts->binary_path);

    /* Wait for child to finish */
    int wstatus;
    waitpid(child, &wstatus, 0);

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* Read counters */
    if (cg_ok == 0) {
        counters_read(&cg, stats);
        stats->wall_seconds = (t1.tv_sec - t0.tv_sec) +
                              (t1.tv_nsec - t0.tv_nsec) / 1e9;
    }

    /* Drain sampling ring buffers */
    if (samp_ok == 0 && cycle_samples)
        sampling_drain(&sctx.cycles, cycle_samples, n_cycle, sa);
    if (cm_samples && samp_ok == 0 && sctx.cache_misses.fd >= 0)
        sampling_drain(&sctx.cache_misses, cm_samples, n_cm, sa);

    /* Read topdown */
    if (td_ok == 0 && topdown_read(&tg, topdown) == 0)
        *has_topdown = 1;

    counters_close(&cg);
    topdown_close(&tg);
    sampling_close(&sctx);

    return 0;
}

/* ── Sample processing — replaces perf report/annotate parsing ───────── */

/* Context markers in callchain arrays */
static int is_context_marker(uint64_t ip) {
    return ip >= (uint64_t)-4096;
}

static int process_samples(struct sym_resolver *sr,
                            struct raw_sample *cycle_samples, int n_cycle,
                            struct raw_sample *cm_samples, int n_cm,
                            struct perf_opts *opts,
                            struct perf_profile *prof) {
    if (!sr->dwfl || n_cycle == 0) return 0;

    /* ── Pass 1: IP→function aggregation (replaces parse_perf_report) ── */
    struct func_bucket {
        char clean_name[256];
        char raw_name[256];
        uint64_t count;
    };

    int n_buckets = 0, cap_buckets = 256;
    struct func_bucket *buckets = xcalloc((size_t)cap_buckets,
                                          sizeof(*buckets));

    for (int i = 0; i < n_cycle; i++) {
        const char *fname = symres_func_name(sr, cycle_samples[i].ip);
        if (!fname) continue;

        char clean[256];
        snprintf(clean, sizeof(clean), "%s", fname);
        strip_compiler_suffix(clean);

        int found = -1;
        for (int j = 0; j < n_buckets; j++) {
            if (strcmp(buckets[j].clean_name, clean) == 0) {
                found = j;
                break;
            }
        }
        if (found >= 0) {
            buckets[found].count++;
        } else {
            if (n_buckets >= cap_buckets) {
                cap_buckets *= 2;
                buckets = xrealloc(buckets,
                    (size_t)cap_buckets * sizeof(*buckets));
            }
            struct func_bucket *b = &buckets[n_buckets++];
            memset(b, 0, sizeof(*b));
            snprintf(b->clean_name, sizeof(b->clean_name), "%s", clean);
            snprintf(b->raw_name, sizeof(b->raw_name), "%s", fname);
            b->count = 1;
        }
    }

    /* Sort by count descending */
    for (int i = 0; i < n_buckets - 1; i++)
        for (int j = i + 1; j < n_buckets; j++)
            if (buckets[j].count > buckets[i].count) {
                struct func_bucket tmp = buckets[i];
                buckets[i] = buckets[j];
                buckets[j] = tmp;
            }

    /* Take top N functions (skip those below 0.5%) */
    int limit = n_buckets < opts->top_n ? n_buckets : opts->top_n;
    int n_total = n_cycle;
    prof->funcs = arena_alloc(&prof->arena, (size_t)limit * sizeof(*prof->funcs));
    prof->n_funcs = 0;

    for (int i = 0; i < limit; i++) {
        double pct = 100.0 * buckets[i].count / n_total;
        if (pct < 0.5) break;
        struct hot_func *hf = &prof->funcs[prof->n_funcs++];
        memset(hf, 0, sizeof(*hf));
        hf->name = (char *)intern_str(&prof->strings, buckets[i].raw_name);
        hf->overhead_pct = pct;
        hf->samples = buckets[i].count;
    }
    free(buckets);

    /* ── Pass 2: Caller extraction (replaces parse_callers) ──────────── */

    /* Per hot-function caller accumulator */
    struct caller_agg {
        char name[256];
        int count;
    };

    int max_callers_per = 32;
    struct arena scratch;
    arena_init(&scratch, ARENA_DEFAULT_BLOCK);
    struct caller_agg *agg_flat = arena_alloc(&scratch,
        (size_t)prof->n_funcs * max_callers_per * sizeof(struct caller_agg));
    memset(agg_flat, 0,
        (size_t)prof->n_funcs * max_callers_per * sizeof(struct caller_agg));
    int *n_agg = arena_alloc(&scratch,
        (size_t)prof->n_funcs * sizeof(int));
    memset(n_agg, 0, (size_t)prof->n_funcs * sizeof(int));

    for (int s = 0; s < n_cycle; s++) {
        if (cycle_samples[s].n_ips < 2) continue;

        const char *self_name = symres_func_name(sr, cycle_samples[s].ip);
        if (!self_name) continue;
        char self_clean[256];
        snprintf(self_clean, sizeof(self_clean), "%s", self_name);
        strip_compiler_suffix(self_clean);

        /* Find which hot function this sample belongs to */
        int func_idx = -1;
        for (int f = 0; f < prof->n_funcs; f++) {
            char fn_clean[256];
            snprintf(fn_clean, sizeof(fn_clean), "%s", prof->funcs[f].name);
            strip_compiler_suffix(fn_clean);
            if (strcmp(self_clean, fn_clean) == 0) {
                func_idx = f;
                break;
            }
        }
        if (func_idx < 0) continue;

        /* Walk callchain: ips[0]=sampled IP, ips[1]=caller, ips[2]=... */
        for (int c = 0; c < cycle_samples[s].n_ips; c++) {
            uint64_t ip = cycle_samples[s].ips[c];
            if (is_context_marker(ip)) continue;

            const char *cname = symres_func_name(sr, ip);
            if (!cname) continue;

            char cclean[256];
            snprintf(cclean, sizeof(cclean), "%s", cname);
            strip_compiler_suffix(cclean);

            /* Skip self and boring callers */
            if (strcmp(cclean, self_clean) == 0) continue;
            if (is_boring_caller(cclean)) continue;
            if (strstr(cname, "@plt")) continue;
            if (strncmp(cname, "__x86_", 6) == 0) continue;

            /* Found a good caller — aggregate */
            int found = -1;
            struct caller_agg *fa = &agg_flat[func_idx * max_callers_per];
            for (int k = 0; k < n_agg[func_idx]; k++) {
                if (strcmp(fa[k].name, cname) == 0) {
                    found = k;
                    break;
                }
            }
            if (found >= 0) {
                fa[found].count++;
            } else if (n_agg[func_idx] < max_callers_per) {
                int idx = n_agg[func_idx]++;
                snprintf(fa[idx].name, sizeof(fa[idx].name), "%s", cname);
                fa[idx].count = 1;
            }
            break; /* take only the first good caller */
        }
    }

    /* Convert aggregated callers to hot_func.callers[] */
    for (int f = 0; f < prof->n_funcs; f++) {
        if (n_agg[f] == 0) continue;
        struct caller_agg *fa = &agg_flat[f * max_callers_per];

        /* Sort by count descending */
        for (int i = 0; i < n_agg[f] - 1; i++)
            for (int j = i + 1; j < n_agg[f]; j++)
                if (fa[j].count > fa[i].count) {
                    struct caller_agg tmp = fa[i];
                    fa[i] = fa[j];
                    fa[j] = tmp;
                }

        /* Compute total samples with callchains for this function */
        int total_cc = 0;
        for (int k = 0; k < n_agg[f]; k++)
            total_cc += fa[k].count;

        /* Take top 5 */
        int nc = n_agg[f] < 5 ? n_agg[f] : 5;
        for (int k = 0; k < nc; k++) {
            double pct = total_cc > 0
                ? 100.0 * fa[k].count / total_cc : 0;
            if (pct < 0.5) break;
            add_caller_entry(prof, &prof->funcs[f], fa[k].name, pct);
        }
    }

    arena_destroy(&scratch);

    /* ── Pass 3: Hot instruction attribution (replaces run_perf_annotate) */

    int insn_cap = 64;
    prof->insns = arena_alloc(&prof->arena, (size_t)insn_cap * sizeof(*prof->insns));
    prof->n_insns = 0;

    struct arena iter_scratch;
    arena_init(&iter_scratch, ARENA_DEFAULT_BLOCK);

    if (sr->cs_ok) {
        for (int f = 0; f < prof->n_funcs; f++) {
            size_t mark = arena_save(&iter_scratch);

            uint64_t func_start;
            uint8_t *func_bytes;
            size_t func_len;

            if (symres_func_range(sr, prof->funcs[f].name,
                                  &func_start, &func_bytes,
                                  &func_len, &iter_scratch) != 0) {
                arena_reset(&iter_scratch, mark);
                continue;
            }

            /* Bucket cycle samples by IP within this function */
            struct { uint64_t ip; int count; } *ip_buckets = NULL;
            int n_ip = 0, cap_ip = 64;
            ip_buckets = arena_alloc(&iter_scratch, (size_t)cap_ip * sizeof(*ip_buckets));
            memset(ip_buckets, 0, (size_t)cap_ip * sizeof(*ip_buckets));
            int func_total = 0;

            for (int s = 0; s < n_cycle; s++) {
                uint64_t ip = cycle_samples[s].ip;
                if (ip >= func_start && ip < func_start + func_len) {
                    func_total++;
                    int found = -1;
                    for (int k = 0; k < n_ip; k++) {
                        if (ip_buckets[k].ip == ip) {
                            found = k;
                            break;
                        }
                    }
                    if (found >= 0) {
                        ip_buckets[found].count++;
                    } else {
                        if (n_ip >= cap_ip) {
                            size_t old_sz = (size_t)cap_ip * sizeof(*ip_buckets);
                            cap_ip *= 2;
                            ip_buckets = arena_realloc(&iter_scratch, ip_buckets,
                                old_sz, (size_t)cap_ip * sizeof(*ip_buckets));
                        }
                        ip_buckets[n_ip].ip = ip;
                        ip_buckets[n_ip].count = 1;
                        n_ip++;
                    }
                }
            }

            /* Sort by count descending */
            for (int i = 0; i < n_ip - 1; i++)
                for (int j = i + 1; j < n_ip; j++)
                    if (ip_buckets[j].count > ip_buckets[i].count) {
                        uint64_t ti = ip_buckets[i].ip;
                        int tc = ip_buckets[i].count;
                        ip_buckets[i].ip = ip_buckets[j].ip;
                        ip_buckets[i].count = ip_buckets[j].count;
                        ip_buckets[j].ip = ti;
                        ip_buckets[j].count = tc;
                    }

            /* Take top insns_n */
            int take = n_ip < opts->insns_n ? n_ip : opts->insns_n;
            for (int k = 0; k < take; k++) {
                double pct = func_total > 0
                    ? 100.0 * ip_buckets[k].count / func_total : 0;
                if (pct < 0.5) break;

                uint64_t ip = ip_buckets[k].ip;
                uint64_t offset = ip - func_start;

                /* Disassemble the instruction at this IP */
                cs_insn *insn;
                size_t cnt = cs_disasm(sr->cs_handle,
                                       func_bytes + offset,
                                       func_len - (size_t)offset,
                                       ip, 1, &insn);
                if (cnt == 0) continue;

                char asm_buf[256];
                snprintf(asm_buf, sizeof(asm_buf), "%s %s",
                         insn[0].mnemonic, insn[0].op_str);
                cs_free(insn, cnt);

                /* Get source line */
                uint32_t src_line = 0;
                const char *src_file = NULL;
                int sline;
                if (symres_srcline(sr, ip, &src_file, &sline) == 0)
                    src_line = (uint32_t)sline;

                if (prof->n_insns >= insn_cap) {
                    size_t old_sz = (size_t)insn_cap * sizeof(*prof->insns);
                    insn_cap *= 2;
                    prof->insns = arena_realloc(&prof->arena, prof->insns,
                        old_sz, (size_t)insn_cap * sizeof(*prof->insns));
                }
                struct hot_insn *hi = &prof->insns[prof->n_insns++];
                hi->func_name = (char *)intern_str(&prof->strings,
                                                   prof->funcs[f].name);
                hi->addr = ip;
                hi->pct = pct;
                hi->asm_text = arena_strdup(&prof->arena, asm_buf);
                hi->source_line = src_line;
                hi->source_file = src_file ?
                    (char *)intern_str(&prof->strings, src_file) : NULL;
            }

            arena_reset(&iter_scratch, mark);
        }
    }

    /* ── Pass 4: Cache-miss attribution (replaces run_cache_misses) ──── */

    if (cm_samples && n_cm > 0 && sr->cs_ok) {
        int cm_cap = 64;
        prof->cm_sites = arena_alloc(&prof->arena, (size_t)cm_cap * sizeof(*prof->cm_sites));
        prof->n_cm_sites = 0;

        for (int f = 0; f < prof->n_funcs; f++) {
            size_t mark = arena_save(&iter_scratch);

            uint64_t func_start;
            uint8_t *func_bytes;
            size_t func_len;

            if (symres_func_range(sr, prof->funcs[f].name,
                                  &func_start, &func_bytes,
                                  &func_len, &iter_scratch) != 0) {
                arena_reset(&iter_scratch, mark);
                continue;
            }

            /* Bucket cache-miss samples by IP, track most common data addr */
            struct { uint64_t ip; uint64_t top_addr; int count; } *ip_buckets = NULL;
            int n_ip = 0, cap_ip = 64;
            ip_buckets = arena_alloc(&iter_scratch, (size_t)cap_ip * sizeof(*ip_buckets));
            memset(ip_buckets, 0, (size_t)cap_ip * sizeof(*ip_buckets));

            for (int s = 0; s < n_cm; s++) {
                uint64_t ip = cm_samples[s].ip;
                if (ip >= func_start && ip < func_start + func_len) {
                    int found = -1;
                    for (int k = 0; k < n_ip; k++) {
                        if (ip_buckets[k].ip == ip) {
                            found = k;
                            break;
                        }
                    }
                    if (found >= 0) {
                        ip_buckets[found].count++;
                        /* Keep first non-zero data addr as representative */
                        if (!ip_buckets[found].top_addr && cm_samples[s].addr)
                            ip_buckets[found].top_addr = cm_samples[s].addr;
                    } else {
                        if (n_ip >= cap_ip) {
                            size_t old_sz = (size_t)cap_ip * sizeof(*ip_buckets);
                            cap_ip *= 2;
                            ip_buckets = arena_realloc(&iter_scratch, ip_buckets,
                                old_sz, (size_t)cap_ip * sizeof(*ip_buckets));
                        }
                        ip_buckets[n_ip].ip = ip;
                        ip_buckets[n_ip].top_addr = cm_samples[s].addr;
                        ip_buckets[n_ip].count = 1;
                        n_ip++;
                    }
                }
            }

            /* Sort by count descending */
            for (int i = 0; i < n_ip - 1; i++)
                for (int j = i + 1; j < n_ip; j++)
                    if (ip_buckets[j].count > ip_buckets[i].count) {
                        uint64_t ti = ip_buckets[i].ip;
                        uint64_t ta = ip_buckets[i].top_addr;
                        int tc = ip_buckets[i].count;
                        ip_buckets[i].ip = ip_buckets[j].ip;
                        ip_buckets[i].top_addr = ip_buckets[j].top_addr;
                        ip_buckets[i].count = ip_buckets[j].count;
                        ip_buckets[j].ip = ti;
                        ip_buckets[j].top_addr = ta;
                        ip_buckets[j].count = tc;
                    }

            int take = n_ip < opts->insns_n ? n_ip : opts->insns_n;
            for (int k = 0; k < take; k++) {
                double pct = n_cm > 0
                    ? 100.0 * ip_buckets[k].count / n_cm : 0;
                if (pct < 0.05) break;

                uint64_t ip = ip_buckets[k].ip;
                uint64_t offset = ip - func_start;

                cs_insn *insn;
                size_t cnt = cs_disasm(sr->cs_handle,
                                       func_bytes + offset,
                                       func_len - (size_t)offset,
                                       ip, 1, &insn);
                if (cnt == 0) continue;

                char asm_buf[256];
                snprintf(asm_buf, sizeof(asm_buf), "%s %s",
                         insn[0].mnemonic, insn[0].op_str);
                cs_free(insn, cnt);

                uint32_t src_line = 0;
                const char *src_file = NULL;
                int sline;
                if (symres_srcline(sr, ip, &src_file, &sline) == 0)
                    src_line = (uint32_t)sline;

                if (prof->n_cm_sites >= cm_cap) {
                    size_t old_sz = (size_t)cm_cap * sizeof(*prof->cm_sites);
                    cm_cap *= 2;
                    prof->cm_sites = arena_realloc(&prof->arena, prof->cm_sites,
                        old_sz, (size_t)cm_cap * sizeof(*prof->cm_sites));
                }
                struct cache_miss_site *cm =
                    &prof->cm_sites[prof->n_cm_sites++];
                cm->func_name = (char *)intern_str(&prof->strings,
                                                   prof->funcs[f].name);
                cm->source_line = src_line;
                cm->pct = pct;
                cm->asm_text = arena_strdup(&prof->arena, asm_buf);
                cm->source_file = src_file ?
                    (char *)intern_str(&prof->strings, src_file) : NULL;
                cm->data_addr = ip_buckets[k].top_addr;
            }

            arena_reset(&iter_scratch, mark);
        }
    }

    /* ── Pass 5: Memory hotspot attribution (PERF_SAMPLE_ADDR) ──────── */

    if (cm_samples && n_cm > 0 && sr->cs_ok) {
        /* Check if any sample has addr != 0 */
        int has_addr = 0;
        for (int i = 0; i < n_cm && !has_addr; i++)
            if (cm_samples[i].addr) has_addr = 1;

        if (has_addr) {
            /* Bucket by (func_name, cache_line = addr >> 6) */
            struct mem_bucket {
                char func[256];
                uint64_t cache_line;
                uint64_t ip;      /* representative IP for disassembly */
                int count;
            };
            int n_mb = 0, cap_mb = 256;
            size_t mbs_mark = arena_save(&iter_scratch);
            struct mem_bucket *mbs = arena_alloc(&iter_scratch, (size_t)cap_mb * sizeof(*mbs));
            memset(mbs, 0, (size_t)cap_mb * sizeof(*mbs));

            for (int i = 0; i < n_cm; i++) {
                if (!cm_samples[i].addr) continue;
                const char *fname = symres_func_name(sr, cm_samples[i].ip);
                if (!fname) continue;

                char clean[256];
                snprintf(clean, sizeof(clean), "%s", fname);
                strip_compiler_suffix(clean);
                uint64_t cl = cm_samples[i].addr >> 6;

                int found = -1;
                for (int j = 0; j < n_mb; j++) {
                    if (mbs[j].cache_line == cl &&
                        strcmp(mbs[j].func, clean) == 0) {
                        found = j;
                        break;
                    }
                }
                if (found >= 0) {
                    mbs[found].count++;
                } else {
                    if (n_mb >= cap_mb) {
                        size_t old_sz = (size_t)cap_mb * sizeof(*mbs);
                        cap_mb *= 2;
                        mbs = arena_realloc(&iter_scratch, mbs,
                            old_sz, (size_t)cap_mb * sizeof(*mbs));
                    }
                    snprintf(mbs[n_mb].func, sizeof(mbs[n_mb].func),
                             "%s", clean);
                    mbs[n_mb].cache_line = cl;
                    mbs[n_mb].ip = cm_samples[i].ip;
                    mbs[n_mb].count = 1;
                    n_mb++;
                }
            }

            /* Count total samples with addr */
            int total_addr = 0;
            for (int i = 0; i < n_mb; i++) total_addr += mbs[i].count;

            /* Sort by count descending */
            for (int i = 0; i < n_mb - 1; i++)
                for (int j = i + 1; j < n_mb; j++)
                    if (mbs[j].count > mbs[i].count) {
                        struct mem_bucket tmp = mbs[i];
                        mbs[i] = mbs[j];
                        mbs[j] = tmp;
                    }

            /* Take top entries */
            int take = n_mb < opts->top_n * 2 ? n_mb : opts->top_n * 2;
            int mh_cap = take > 0 ? take : 1;
            prof->mem_hotspots = arena_alloc(&prof->arena, (size_t)mh_cap *
                                        sizeof(*prof->mem_hotspots));
            prof->n_mem_hotspots = 0;

            for (int k = 0; k < take; k++) {
                double pct = total_addr > 0
                    ? 100.0 * mbs[k].count / total_addr : 0;
                if (pct < 0.1) break;

                uint64_t ip = mbs[k].ip;

                /* Disassemble */
                char asm_buf[256];
                asm_buf[0] = '\0';

                /* Find func range to disassemble */
                uint64_t func_start;
                uint8_t *func_bytes;
                size_t func_len;
                size_t fb_mark = arena_save(&iter_scratch);
                if (symres_func_range(sr, mbs[k].func,
                                      &func_start, &func_bytes,
                                      &func_len, &iter_scratch) == 0) {
                    uint64_t offset = ip - func_start;
                    if (offset < func_len) {
                        cs_insn *dinsn;
                        size_t dcnt = cs_disasm(sr->cs_handle,
                                                func_bytes + offset,
                                                func_len - (size_t)offset,
                                                ip, 1, &dinsn);
                        if (dcnt > 0) {
                            snprintf(asm_buf, sizeof(asm_buf), "%s %s",
                                     dinsn[0].mnemonic, dinsn[0].op_str);
                            cs_free(dinsn, dcnt);
                        }
                    }
                }
                arena_reset(&iter_scratch, fb_mark);

                uint32_t src_line = 0;
                const char *src_file = NULL;
                int sline;
                if (symres_srcline(sr, ip, &src_file, &sline) == 0)
                    src_line = (uint32_t)sline;

                struct mem_hotspot *mh =
                    &prof->mem_hotspots[prof->n_mem_hotspots++];
                mh->func_name = (char *)intern_str(&prof->strings,
                                                   mbs[k].func);
                mh->source_line = src_line;
                mh->source_file = src_file ?
                    (char *)intern_str(&prof->strings, src_file) : NULL;
                mh->asm_text = arena_strdup(&prof->arena, asm_buf);
                mh->cache_line = mbs[k].cache_line;
                mh->pct = pct;
                mh->n_samples = mbs[k].count;
            }

            arena_reset(&iter_scratch, mbs_mark);
        }
    }

    arena_destroy(&iter_scratch);
    return 0;
}

/* ── AMDuProf (unchanged — vendor subprocess) ────────────────────────── */

static int run_uprof(struct perf_opts *opts, struct perf_profile *prof) {
    if (opts->uprof_mode < 0) return 0;
    const char *uprof_bin = find_tool("AMDuProfCLI");
    if (!uprof_bin) {
        if (opts->uprof_mode > 0) {
            fprintf(stderr,
                "error: --uprof specified but AMDuProfCLI not found\n");
            return -1;
        }
        return 0;
    }

    char uprof_dir[PATH_MAX];
    snprintf(uprof_dir, sizeof(uprof_dir), "%s/uprof", g_tmpdir);

    /* Phase 1: collect with assess config */
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "'%s' collect --config assess -o '%s' %s 2>&1",
        uprof_bin, uprof_dir, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);

    char data_dir[PATH_MAX] = {0};
    if (out) {
        const char *tag = "Generated data files path: ";
        char *p = strstr(out, tag);
        if (p) {
            p += strlen(tag);
            char *nl = strchr(p, '\n');
            size_t len = nl ? (size_t)(nl - p) : strlen(p);
            while (len > 0 && (p[len-1] == '\r' || p[len-1] == ' '))
                len--;
            if (len < sizeof(data_dir)) {
                memcpy(data_dir, p, len);
                data_dir[len] = '\0';
            }
        }
    }
    free(out);

    if (status != 0 || !data_dir[0]) {
        if (opts->uprof_mode > 0)
            fprintf(stderr, "AMDuProfCLI collect failed\n");
        return opts->uprof_mode > 0 ? -1 : 0;
    }

    /* Phase 2: generate report */
    snprintf(cmd, sizeof(cmd),
        "'%s' report -i '%s' --category cpu 2>&1", uprof_bin, data_dir);
    out = run_cmd(cmd, &status, opts->verbose);
    free(out);
    if (status != 0) return 0;

    /* Phase 3: read and parse report.csv */
    char csv_path[PATH_MAX];
    snprintf(csv_path, sizeof(csv_path), "%s/report.csv", data_dir);

    FILE *fp = fopen(csv_path, "r");
    if (!fp) return 0;

    int cap = prof->n_funcs > 0 ? prof->n_funcs : 1;
    prof->uprof_funcs = arena_alloc(&prof->arena, (size_t)cap * sizeof(*prof->uprof_funcs));
    prof->n_uprof_funcs = 0;

    char line[4096];
    int in_funcs = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "HOTTEST FUNCTIONS")) {
            if (!fgets(line, sizeof(line), fp)) break;
            in_funcs = 1;
            continue;
        }
        if (!in_funcs) continue;

        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            break;

        if (line[0] != '"') break;
        char *name_end = strchr(line + 1, '"');
        if (!name_end) continue;
        *name_end = '\0';
        const char *func_name = line + 1;

        int matched = 0;
        for (int f = 0; f < prof->n_funcs; f++) {
            if (strcmp(func_name, prof->funcs[f].name) == 0) {
                matched = 1;
                break;
            }
        }
        if (!matched) continue;

        char *p = name_end + 1;
        double cycles_val, inst_val, ipc, cpi_unused;
        double br_misp_pti = 0, br_misp_pct = 0;
        double l1_dc_acc_pti = 0, l1_dc_miss_pti = 0, l1_dc_miss_pct = 0;
        double l1_refill_local = 0, l1_refill_remote = 0;
        double misaligned_pti = 0;

        if (sscanf(p,
            ",%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,",
            &cycles_val, &inst_val, &ipc, &cpi_unused,
            &br_misp_pti, &br_misp_pct,
            &l1_dc_acc_pti, &l1_dc_miss_pti, &l1_dc_miss_pct,
            &l1_refill_local, &l1_refill_remote,
            &misaligned_pti) < 4)
            continue;

        struct uprof_func *uf =
            &prof->uprof_funcs[prof->n_uprof_funcs++];
        uf->name = (char *)intern_str(&prof->strings, func_name);
        uf->ipc = ipc;
        uf->l1_dc_miss_pct = l1_dc_miss_pct;
        uf->br_misp_pti = br_misp_pti;
        uf->misaligned_pti = misaligned_pti;
    }
    fclose(fp);
    return 0;
}

/* ── llvm-mca throughput — capstone replaces objdump ─────────────────── */

static int run_mca(struct sym_resolver *sr, struct perf_opts *opts,
                    struct perf_profile *prof) {
    if (opts->mca_mode < 0) return 0;
    if (!has_tool("llvm-mca")) {
        if (opts->mca_mode > 0) {
            fprintf(stderr,
                "error: --mca specified but llvm-mca not found\n");
            return -1;
        }
        return 0;
    }
    if (!sr->cs_ok) return 0;

    prof->n_mca_blocks = 0;

    /* Check if -mcpu=native works */
    int status;
    char *probe = run_cmd(
        "echo 'nop' | llvm-mca -mcpu=native 2>&1", &status, 0);
    int has_native = (status == 0);
    free(probe);
    const char *mcpu_flag = has_native ? "-mcpu=native" : "";

    int limit = prof->n_funcs < opts->top_n ?
                prof->n_funcs : opts->top_n;
    int mca_cap = limit > 0 ? limit : 1;
    prof->mca_blocks = arena_alloc(&prof->arena, (size_t)mca_cap * sizeof(*prof->mca_blocks));

    struct arena mca_scratch;
    arena_init(&mca_scratch, ARENA_DEFAULT_BLOCK);

    for (int f = 0; f < limit; f++) {
        const char *raw_name = prof->funcs[f].name;

        /* Get function bytes via sym_resolver */
        size_t mark = arena_save(&mca_scratch);
        uint64_t func_start;
        uint8_t *func_bytes;
        size_t func_len;
        if (symres_func_range(sr, raw_name,
                              &func_start, &func_bytes, &func_len,
                              &mca_scratch) != 0) {
            arena_reset(&mca_scratch, mark);
            continue;
        }

        /* Disassemble with capstone */
        cs_insn *insns;
        size_t count = cs_disasm(sr->cs_handle, func_bytes, func_len,
                                  func_start, 0, &insns);
        arena_reset(&mca_scratch, mark);
        if (count == 0) continue;

        /* Write cleaned instructions to temp file */
        char tmpasm[PATH_MAX];
        snprintf(tmpasm, sizeof(tmpasm), "%s/mca_block.s", g_tmpdir);
        FILE *fp = fopen(tmpasm, "w");
        if (!fp) { cs_free(insns, count); continue; }

        int n_written = 0;
        for (size_t i = 0; i < count; i++) {
            /* Skip endbr, nops */
            if (strcmp(insns[i].mnemonic, "endbr64") == 0 ||
                strcmp(insns[i].mnemonic, "endbr32") == 0)
                continue;

            /* For branches/calls, replace hex targets with labels */
            if (insns[i].mnemonic[0] == 'j' ||
                strncmp(insns[i].mnemonic, "call", 4) == 0) {
                const char *op = insns[i].op_str;
                /* Check if operand is a bare hex address (0x...) */
                if (op[0] == '0' && op[1] == 'x') {
                    fprintf(fp, "%s .Ltmp\n", insns[i].mnemonic);
                    n_written++;
                    continue;
                }
                /* Check if operand is a bare hex number (no prefix) */
                int all_hex = (*op != '\0');
                for (const char *c = op; *c; c++) {
                    if (!((*c >= '0' && *c <= '9') ||
                          (*c >= 'a' && *c <= 'f') ||
                          (*c >= 'A' && *c <= 'F'))) {
                        all_hex = 0;
                        break;
                    }
                }
                if (all_hex && *op) {
                    fprintf(fp, "%s .Ltmp\n", insns[i].mnemonic);
                    n_written++;
                    continue;
                }
            }

            /* Strip comments after # */
            char cleaned[512];
            snprintf(cleaned, sizeof(cleaned), "%s %s",
                     insns[i].mnemonic, insns[i].op_str);
            char *hash = strchr(cleaned, '#');
            if (hash) {
                *hash = '\0';
                /* trim trailing spaces */
                size_t cl = strlen(cleaned);
                while (cl > 0 && cleaned[cl-1] == ' ')
                    cleaned[--cl] = '\0';
            }

            if (cleaned[0]) {
                fprintf(fp, "%s\n", cleaned);
                n_written++;
            }
        }
        cs_free(insns, count);
        fclose(fp);

        if (n_written == 0) continue;

        /* Run llvm-mca */
        char cmd[4096];
        snprintf(cmd, sizeof(cmd),
            "llvm-mca %s -iterations=100 --timeline=0 "
            "--all-stats < '%s' 2>&1",
            mcpu_flag, tmpasm);

        char *mca_out = run_cmd(cmd, &status, opts->verbose);
        if (!mca_out || status != 0) { free(mca_out); continue; }

        /* Parse: Block RThroughput, Total uOps, IPC */
        double rthroughput = 0, ipc = 0;
        int uops = 0;
        char bottleneck[128] = "";

        char *line = mca_out;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            char *colon;
            if (strstr(line, "Block RThroughput") &&
                    (colon = strchr(line, ':')))
                sscanf(colon + 1, " %lf", &rthroughput);
            else if (strstr(line, "Total uOps") &&
                    (colon = strchr(line, ':')))
                sscanf(colon + 1, " %d", &uops);
            else if (strstr(line, "IPC") && !strstr(line, "Block") &&
                    (colon = strchr(line, ':')))
                sscanf(colon + 1, " %lf", &ipc);
            else if (strstr(line, "Bottleneck")) {
                char *p = strchr(line, ':');
                if (p) {
                    p++;
                    while (*p == ' ') p++;
                    snprintf(bottleneck, sizeof(bottleneck), "%s", p);
                    size_t blen = strlen(bottleneck);
                    while (blen > 0 &&
                           (bottleneck[blen-1] == ' ' ||
                            bottleneck[blen-1] == '\n'))
                        bottleneck[--blen] = '\0';
                }
            }

            line = nl ? nl + 1 : NULL;
        }
        free(mca_out);

        if (rthroughput > 0 || uops > 0) {
            struct mca_block *mb =
                &prof->mca_blocks[prof->n_mca_blocks++];
            mb->func_name = (char *)intern_str(&prof->strings,
                                               prof->funcs[f].name);
            mb->block_rthroughput = rthroughput;
            mb->ipc = ipc;
            mb->n_uops = uops;
            mb->bottleneck = bottleneck[0] ?
                arena_strdup(&prof->arena, bottleneck) : NULL;
        }
    }

    arena_destroy(&mca_scratch);
    return 0;
}

/* ── pahole struct layout (unchanged — external tool) ────────────────── */

static int run_pahole(struct perf_opts *opts, struct perf_profile *prof,
                      int has_debug) {
    if (opts->pahole_mode < 0) return 0;
    if (!has_tool("pahole")) {
        if (opts->pahole_mode > 0) {
            fprintf(stderr,
                "error: --pahole specified but pahole not found\n");
            return -1;
        }
        return 0;
    }
    if (!has_debug) {
        if (opts->pahole_mode > 0)
            fprintf(stderr,
                "warning: --pahole requires debug symbols (-g)\n");
        return 0;
    }

    int cap = 16;
    prof->layouts = arena_alloc(&prof->arena, (size_t)cap * sizeof(*prof->layouts));
    prof->n_layouts = 0;

    /* Skip known primitive types when probing pahole */
    static const char *skip_types[] = {
        "void", "char", "int", "long", "short", "float", "double",
        "unsigned", "signed", "const", "restrict", "volatile",
        "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "FILE", "bool", "_Bool", NULL,
    };

    /* For each hot function, extract pointer parameter types */
    for (int f = 0; f < prof->n_funcs; f++) {
        const char *sig = prof->funcs[f].skeleton_sig;
        if (!sig) continue;

        const char *p = sig;
        while (*p) {
            while (*p && !(*p >= 'a' && *p <= 'z') &&
                   !(*p >= 'A' && *p <= 'Z') && *p != '_')
                p++;
            if (!*p) break;

            char type_name[256];
            int ti = 0;
            while (*p && ((*p >= 'a' && *p <= 'z') ||
                          (*p >= 'A' && *p <= 'Z') ||
                          (*p >= '0' && *p <= '9') || *p == '_') &&
                   ti < (int)sizeof(type_name) - 1)
                type_name[ti++] = *p++;
            type_name[ti] = '\0';

            const char *q = p;
            while (*q == ' ') q++;
            if (*q != '*') continue;

            int skip = 0;
            for (int s = 0; skip_types[s]; s++) {
                if (strcmp(type_name, skip_types[s]) == 0) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            const char *probe_name = type_name;
            if (strcmp(type_name, "struct") == 0) {
                while (*p == ' ') p++;
                ti = 0;
                while (*p && ((*p >= 'a' && *p <= 'z') ||
                              (*p >= 'A' && *p <= 'Z') ||
                              (*p >= '0' && *p <= '9') || *p == '_') &&
                       ti < (int)sizeof(type_name) - 1)
                    type_name[ti++] = *p++;
                type_name[ti] = '\0';
                probe_name = type_name;
            }

            if (!*probe_name) continue;

            int dup = 0;
            for (int i = 0; i < prof->n_layouts; i++) {
                if (strcmp(prof->layouts[i].type_name,
                           probe_name) == 0) {
                    dup = 1;
                    break;
                }
            }
            if (dup) continue;

            char cmd[4096];
            snprintf(cmd, sizeof(cmd),
                "pahole -C '%s' '%s' 2>/dev/null",
                probe_name, opts->binary_path);

            int status;
            char *out = run_cmd(cmd, &status, opts->verbose);
            if (!out || status != 0 || !*out) {
                free(out);
                continue;
            }

            uint32_t size = 0, cachelines = 0;
            uint32_t holes = 0, padding = 0;

            char *line = out;
            while (line && *line) {
                char *nl = strchr(line, '\n');
                if (nl) *nl = '\0';

                if (strstr(line, "/* size:")) {
                    char *sp = strstr(line, "size:");
                    if (sp) sscanf(sp, "size: %u", &size);
                    sp = strstr(line, "cachelines:");
                    if (sp) sscanf(sp, "cachelines: %u", &cachelines);
                }
                if (strstr(line, "holes:")) {
                    char *sp = strstr(line, "holes:");
                    if (sp) sscanf(sp, "holes: %u", &holes);
                }
                if (strstr(line, "padding:")) {
                    char *sp = strstr(line, "padding:");
                    if (sp) sscanf(sp, "padding: %u", &padding);
                }

                line = nl ? nl + 1 : NULL;
            }
            free(out);

            if (holes == 0 && cachelines <= 1) continue;

            if (prof->n_layouts >= cap) {
                size_t old_sz = (size_t)cap * sizeof(*prof->layouts);
                cap *= 2;
                prof->layouts = arena_realloc(&prof->arena, prof->layouts,
                    old_sz, (size_t)cap * sizeof(*prof->layouts));
            }
            struct struct_layout *sl =
                &prof->layouts[prof->n_layouts++];
            sl->type_name = (char *)intern_str(&prof->strings, probe_name);
            sl->size = size;
            sl->holes = holes;
            sl->padding = padding;
            sl->cachelines = cachelines;
            sl->func_name = (char *)intern_str(&prof->strings,
                                               prof->funcs[f].name);
        }
    }

    return 0;
}

/* ── Skeleton cross-reference (unchanged) ────────────────────────────── */

static void xref_skeleton(struct perf_opts *opts,
                          struct perf_profile *prof) {
    opt_show_calls = 1;

    TSParser *parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_c());

    uint32_t err_offset;
    TSQueryError err_type;
    TSQuery *query = ts_query_new(
        tree_sitter_c(), QUERY_SOURCE,
        (uint32_t)strlen(QUERY_SOURCE),
        &err_offset, &err_type);
    if (!query) {
        ts_parser_delete(parser);
        return;
    }

    /* Walk source directory with cache */
    char *xref_opts = cache_make_opts_str(1, NULL, 0, NULL, 0, NULL, 0);
    archmap_cache *xref_cache = cache_open(opts->source_dir, xref_opts);
    free(xref_opts);

    char *src_dup = strdup(opts->source_dir);
    char *paths[] = { src_dup, NULL };
    FTS *ftsp = fts_open(paths, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
    if (ftsp) {
        FTSENT *ent;
        while ((ent = fts_read(ftsp))) {
            if (should_skip_fts_entry(ent)) {
                fts_set(ftsp, ent, FTS_SKIP);
                continue;
            }
            if (ent->fts_info == FTS_F &&
                is_source_file(ent->fts_name)) {
                int hit = 0;
                if (xref_cache) {
                    struct file_entry tmp;
                    char **tags; int n_tags;
                    if (cache_lookup(xref_cache, ent->fts_path,
                                     &tmp, &tags, &n_tags) == 1) {
                        struct file_entry *fe = add_file(ent->fts_path);
                        free(fe->abs_path);
                        *fe = tmp;
                        for (int t = 0; t < n_tags; t++) free(tags[t]);
                        free(tags);
                        hit = 1;
                    }
                }
                if (!hit) {
                    g_n_file_tags = 0;
                    collect_file(ent->fts_path, parser, query,
                                 NULL, 0, NULL, 0);
                    if (xref_cache && g_n_files > 0)
                        cache_store(xref_cache, ent->fts_path,
                                    &g_files[g_n_files - 1],
                                    g_file_tags, g_n_file_tags);
                    for (int t = 0; t < g_n_file_tags; t++)
                        free(g_file_tags[t]);
                    g_n_file_tags = 0;
                }
            }
        }
        fts_close(ftsp);
    }
    free(src_dup);

    if (xref_cache) cache_close(xref_cache);

    /* Match hot functions → collected symbols */
    const char *bin_base = opts->binary_path ?
        strrchr(opts->binary_path, '/') : NULL;
    bin_base = bin_base ? bin_base + 1 :
               (opts->binary_path ? opts->binary_path : "");

    for (int f = 0; f < prof->n_funcs; f++) {
        struct hot_func *hf = &prof->funcs[f];
        char clean[256];
        snprintf(clean, sizeof(clean), "%s", hf->name);
        strip_compiler_suffix(clean);
        size_t clen = strlen(clean);

        struct file_entry *best_fe = NULL;
        struct symbol *best_sym = NULL;
        int best_score = -1;

        for (int i = 0; i < g_n_files; i++) {
            struct file_entry *fe = &g_files[i];
            for (int j = 0; j < fe->n_syms; j++) {
                struct symbol *sym = &fe->syms[j];
                if (!sym->text) continue;

                char *pos = strstr(sym->text, clean);
                if (!pos) continue;

                /* Word-boundary check */
                if (pos > sym->text) {
                    char prev = pos[-1];
                    if (prev != ' ' && prev != '*' &&
                        prev != '\n' && prev != '\t')
                        continue;
                }
                char after = pos[clen];
                if (after != '(' && after != ' ' &&
                    after != ';' && after != '\0')
                    continue;

                int score = 0;
                if (*bin_base) {
                    const char *src_base = strrchr(fe->abs_path, '/');
                    src_base = src_base ? src_base + 1 : fe->abs_path;
                    char src_stem[256];
                    snprintf(src_stem, sizeof(src_stem), "%s", src_base);
                    char *dot = strrchr(src_stem, '.');
                    if (dot) *dot = '\0';
                    if (strstr(bin_base, src_stem))
                        score = 2;
                }

                if (score > best_score) {
                    best_fe = fe;
                    best_sym = sym;
                    best_score = score;
                }
            }
        }

        if (best_sym) {
            hf->skeleton_sig = arena_strdup(&prof->arena, best_sym->text);
            hf->source_file = (char *)intern_str(&prof->strings,
                                                  best_fe->abs_path);
            hf->start_line = best_sym->start_line;
            hf->end_line = best_sym->end_line;

            if (best_sym->n_callees > 0) {
                hf->n_callees = best_sym->n_callees;
                hf->callees = arena_alloc(&prof->arena,
                    (size_t)best_sym->n_callees * sizeof(char *));
                for (int k = 0; k < best_sym->n_callees; k++)
                    hf->callees[k] =
                        (char *)intern_str(&prof->strings,
                                           best_sym->callees[k]);
            }
        }
    }

    ts_query_delete(query);
    ts_parser_delete(parser);
}

/* ── Compiler optimization remarks (unchanged — subprocess) ──────────── */

static int run_remarks(struct perf_opts *opts, struct perf_profile *prof) {
    if (opts->remarks_mode < 0) return 0;

    /* Detect compiler */
    int status;
    char *cc_out = run_cmd("cc --version 2>&1", &status, 0);
    int is_gcc = 0, is_clang = 0;
    if (cc_out) {
        for (char *p = cc_out; *p; p++)
            *p = (*p >= 'A' && *p <= 'Z') ? *p + 32 : *p;
        if (strstr(cc_out, "gcc") || strstr(cc_out, "g++"))
            is_gcc = 1;
        else if (strstr(cc_out, "clang"))
            is_clang = 1;
        free(cc_out);
    }
    if (!is_gcc && !is_clang) {
        if (opts->remarks_mode > 0)
            fprintf(stderr, "error: --remarks requires gcc or clang as cc\n");
        return opts->remarks_mode > 0 ? -1 : 0;
    }

    /* Collect unique source files from hot functions */
    char *src_files[64];
    int n_src = 0;
    for (int f = 0; f < prof->n_funcs && n_src < 64; f++) {
        if (!prof->funcs[f].source_file) continue;
        int dup = 0;
        for (int s = 0; s < n_src; s++) {
            if (strcmp(src_files[s], prof->funcs[f].source_file) == 0) {
                dup = 1; break;
            }
        }
        if (!dup)
            src_files[n_src++] = prof->funcs[f].source_file;
    }
    if (n_src == 0) return 0;

    int remark_cap = prof->n_funcs * 10;
    prof->remarks = arena_alloc(&prof->arena, (size_t)remark_cap * sizeof(*prof->remarks));
    prof->n_remarks = 0;

    /* Try to read compile_commands.json */
    char ccjson_path[PATH_MAX];
    snprintf(ccjson_path, sizeof(ccjson_path), "%s/compile_commands.json",
             opts->source_dir);
    char *ccjson = NULL;
    {
        FILE *fp = fopen(ccjson_path, "r");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long flen = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (flen > 0 && flen < 10*1024*1024) {
                ccjson = xmalloc((size_t)flen + 1);
                size_t nread = fread(ccjson, 1, (size_t)flen, fp);
                ccjson[nread] = '\0';
            }
            fclose(fp);
        }
    }

    for (int s = 0; s < n_src; s++) {
        char cmd[8192];
        int found_cmd = 0;

        if (ccjson) {
            const char *basename = strrchr(src_files[s], '/');
            basename = basename ? basename + 1 : src_files[s];

            char needle[PATH_MAX];
            snprintf(needle, sizeof(needle), "\"file\":\"%s\"", basename);
            char *entry = strstr(ccjson, needle);
            if (!entry) {
                snprintf(needle, sizeof(needle), "\"%s\"", basename);
                char *p = ccjson;
                while ((p = strstr(p, "\"file\"")) != NULL) {
                    char *colon = strchr(p + 6, ':');
                    if (!colon) break;
                    char *q = colon + 1;
                    while (*q == ' ') q++;
                    if (*q == '"') {
                        q++;
                        char *end = strchr(q, '"');
                        if (end) {
                            const char *fn = end;
                            while (fn > q && fn[-1] != '/') fn--;
                            if ((size_t)(end - fn) == strlen(basename) &&
                                strncmp(fn, basename, strlen(basename)) == 0) {
                                entry = p;
                                break;
                            }
                        }
                    }
                    p += 6;
                }
            }
            if (entry) {
                char *cmd_field = NULL;
                char *search = entry;
                for (int tries = 0; tries < 2000 && *search; tries++, search++) {
                    if (strncmp(search, "\"command\"", 9) == 0) {
                        cmd_field = search;
                        break;
                    }
                }
                if (!cmd_field) {
                    search = entry;
                    for (int tries = 0; tries < 2000 && search > ccjson;
                         tries++, search--) {
                        if (strncmp(search, "\"command\"", 9) == 0) {
                            cmd_field = search;
                            break;
                        }
                    }
                }
                if (cmd_field) {
                    char *colon = strchr(cmd_field + 9, ':');
                    if (colon) {
                        char *q = colon + 1;
                        while (*q == ' ') q++;
                        if (*q == '"') {
                            q++;
                            char orig_cmd[4096];
                            int ci = 0;
                            while (*q && *q != '"' &&
                                   ci < (int)sizeof(orig_cmd) - 1) {
                                if (*q == '\\' && q[1]) { q++; }
                                orig_cmd[ci++] = *q++;
                            }
                            orig_cmd[ci] = '\0';
                            const char *rflags = is_gcc ?
                                "-fopt-info-optimized-missed" :
                                "-Rpass=.* -Rpass-missed=.*";
                            snprintf(cmd, sizeof(cmd),
                                "%s -S -o /dev/null %s 2>&1",
                                orig_cmd, rflags);
                            found_cmd = 1;
                        }
                    }
                }
            }
        }

        if (!found_cmd) {
            const char *rflags = is_gcc ?
                "-fopt-info-optimized-missed" :
                "-Rpass=.* -Rpass-missed=.*";
            snprintf(cmd, sizeof(cmd),
                "cc -S -o /dev/null -O3 -march=native -I'%s' '%s' %s 2>&1",
                opts->source_dir, src_files[s], rflags);
        }

        char *out = run_cmd(cmd, &status, opts->verbose);
        if (!out) continue;

        char *line = out;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            char rfile[PATH_MAX];
            uint32_t rline;
            uint32_t rcol;
            char category[64];
            char message[512];

            int matched = 0;
            if (sscanf(line, "%[^:]:%u:%u: %63[^:]: %511[^\n]",
                       rfile, &rline, &rcol, category, message) == 5)
                matched = 1;

            if (matched) {
                char *cat = category;
                while (*cat == ' ') cat++;
                size_t catlen = strlen(cat);
                while (catlen > 0 && cat[catlen-1] == ' ')
                    cat[--catlen] = '\0';

                const char *norm_cat = NULL;
                if (strstr(cat, "optimized") || strstr(cat, "remark"))
                    norm_cat = "optimized";
                else if (strstr(cat, "missed") || strstr(cat, "note"))
                    norm_cat = "missed";
                else {
                    line = nl ? nl + 1 : NULL;
                    continue;
                }

                for (int f = 0; f < prof->n_funcs; f++) {
                    struct hot_func *hf = &prof->funcs[f];
                    if (!hf->source_file) continue;

                    const char *hf_base = strrchr(hf->source_file, '/');
                    hf_base = hf_base ? hf_base + 1 : hf->source_file;
                    const char *rf_base = strrchr(rfile, '/');
                    rf_base = rf_base ? rf_base + 1 : rfile;
                    if (strcmp(hf_base, rf_base) != 0) continue;

                    if (rline < hf->start_line || rline > hf->end_line)
                        continue;

                    int func_remarks = 0;
                    for (int r = 0; r < prof->n_remarks; r++) {
                        if (strcmp(prof->remarks[r].func_name,
                                   hf->name) == 0)
                            func_remarks++;
                    }
                    if (func_remarks >= 10) break;

                    struct remark_entry *re =
                        &prof->remarks[prof->n_remarks++];
                    re->func_name = (char *)intern_str(&prof->strings,
                                                       hf->name);
                    re->source_file = (char *)intern_str(&prof->strings,
                                                         hf->source_file);
                    re->line = rline;
                    re->category = (char *)intern_str(&prof->strings,
                                                      norm_cat);
                    re->message = arena_strdup(&prof->arena, message);
                    break;
                }
            }

            line = nl ? nl + 1 : NULL;
        }
        free(out);
    }
    free(ccjson);

    /* Sort: "missed" before "optimized" within each function */
    for (int i = 0; i < prof->n_remarks - 1; i++)
        for (int j = i + 1; j < prof->n_remarks; j++) {
            int same_func = strcmp(prof->remarks[i].func_name,
                                   prof->remarks[j].func_name) == 0;
            if (same_func &&
                strcmp(prof->remarks[i].category, "optimized") == 0 &&
                strcmp(prof->remarks[j].category, "missed") == 0) {
                struct remark_entry tmp = prof->remarks[i];
                prof->remarks[i] = prof->remarks[j];
                prof->remarks[j] = tmp;
            }
        }

    return 0;
}

/* ── Source interleaving helpers ──────────────────────────────────────── */

/* Returns array where lines[i] points to start of line i (1-indexed).
   lines[0] = NULL. Caller frees the array (not the strings — they point
   into src). *n_lines is set to the highest valid line number. */
static char **index_source_lines(const char *src, long len, int *n_lines) {
    int cap = 256, n = 0;
    char **lines = xcalloc((size_t)cap, sizeof(char *));
    const char *p = src, *end = src + len;
    while (p < end) {
        n++;
        if (n >= cap) { cap *= 2; lines = xrealloc(lines, (size_t)cap * sizeof(char *)); }
        lines[n] = (char *)p;
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        p = nl ? nl + 1 : end;
    }
    *n_lines = n;
    return lines;
}

/* Print source line with leading whitespace trimmed, no trailing newline */
static void print_source_line(const char *line) {
    /* skip leading whitespace */
    while (*line == ' ' || *line == '\t') line++;
    /* print until newline or NUL */
    while (*line && *line != '\n' && *line != '\r') {
        putchar(*line);
        line++;
    }
}

/* ── Statistics ───────────────────────────────────────────────────────── */

static void compute_stats(struct run_stats *rs) {
    if (rs->n <= 0) { rs->mean = rs->stddev = 0; return; }
    double sum = 0;
    for (int i = 0; i < rs->n; i++) sum += rs->values[i];
    rs->mean = sum / rs->n;
    if (rs->n < 2) { rs->stddev = 0; return; }
    double ss = 0;
    for (int i = 0; i < rs->n; i++) {
        double d = rs->values[i] - rs->mean;
        ss += d * d;
    }
    rs->stddev = sqrt(ss / (rs->n - 1));
}

/* Continued fraction for regularized incomplete beta (Numerical Recipes). */
static double betacf(double a, double b, double x) {
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 200; m++) {
        int m2 = 2 * m;
        /* Even step */
        double aa = (double)m * (b - (double)m) * x /
                    ((qam + (double)m2) * (a + (double)m2));
        d = 1.0 + aa * d; if (fabs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c; if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        h *= d * c;
        /* Odd step */
        aa = -((a + (double)m) * (qab + (double)m) * x) /
              ((a + (double)m2) * (qap + (double)m2));
        d = 1.0 + aa * d; if (fabs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c; if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < 1e-14) break;
    }
    return h;
}

/* Regularized incomplete beta function I_x(a, b). */
static double ibeta(double x, double a, double b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    double bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
                    a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    else
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

/* Student's t CDF via regularized incomplete beta function.
   CDF(t, df) = 1 - 0.5 * I_x(df/2, 0.5) where x = df/(df + t^2). */
static double t_cdf(double t_val, double df) {
    double x = df / (df + t_val * t_val);
    double ix = ibeta(x, df / 2.0, 0.5);

    if (t_val >= 0)
        return 1.0 - 0.5 * ix;
    else
        return 0.5 * ix;
}

static double welch_t_test(struct run_stats *a, struct run_stats *b) {
    if (a->n < 2 || b->n < 2) return 1.0;
    double va = a->stddev * a->stddev, vb = b->stddev * b->stddev;
    double se = sqrt(va / a->n + vb / b->n);
    if (se < 1e-15) return (fabs(a->mean - b->mean) < 1e-15) ? 1.0 : 0.0;
    double t_val = (a->mean - b->mean) / se;

    /* Welch-Satterthwaite degrees of freedom */
    double num = (va / a->n + vb / b->n);
    num *= num;
    double denom = (va * va) / ((double)a->n * a->n * (a->n - 1)) +
                   (vb * vb) / ((double)b->n * b->n * (b->n - 1));
    double df = denom > 0 ? num / denom : 2.0;

    /* Two-sided p-value */
    double p = 2.0 * (1.0 - t_cdf(fabs(t_val), df));
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    return p;
}

static const char *sig_marker(double p) {
    if (p < 0.001) return "***";
    if (p < 0.01)  return "**";
    if (p < 0.05)  return "*";
    return "ns";
}

static void fmt_stddev(char *buf, size_t sz, double stddev) {
    if (stddev >= 1e9)
        snprintf(buf, sz, "%.2fG", stddev / 1e9);
    else if (stddev >= 1e6)
        snprintf(buf, sz, "%.1fM", stddev / 1e6);
    else if (stddev >= 1e3)
        snprintf(buf, sz, "%.0fK", stddev / 1e3);
    else if (stddev >= 1.0)
        snprintf(buf, sz, "%.0f", stddev);
    else
        snprintf(buf, sz, "%.2f", stddev);
}

/* ── Output ──────────────────────────────────────────────────────────── */

static const char *short_path(const char *path,
                              const char *source_dir) {
    if (!path) return "??";
    char real_src[PATH_MAX];
    if (realpath(source_dir, real_src)) {
        size_t len = strlen(real_src);
        if (strncmp(path, real_src, len) == 0 &&
            path[len] == '/')
            return path + len + 1;
    }
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static void print_report(struct perf_opts *opts,
                          struct perf_profile *prof) {
    struct perf_stats *s = &prof->stats;
    char c_cyc[32], c_insn[32], c_cr[32], c_cm[32], c_br[32], c_bm[32];

    fmt_count(c_cyc,  sizeof(c_cyc),  s->cycles);
    fmt_count(c_insn, sizeof(c_insn), s->instructions);
    fmt_count(c_cr,   sizeof(c_cr),   s->cache_refs);
    fmt_count(c_cm,   sizeof(c_cm),   s->cache_misses);
    fmt_count(c_br,   sizeof(c_br),   s->branches);
    fmt_count(c_bm,   sizeof(c_bm),   s->branch_misses);

    printf("=== perf: %s ===\n", opts->cmd_str);
    if (prof->n_runs > 1) {
        char sd_cyc[32], sd_insn[32], sd_ipc[32], sd_wall[32];
        fmt_stddev(sd_cyc,  sizeof(sd_cyc),  prof->rs_cycles.stddev);
        fmt_stddev(sd_insn, sizeof(sd_insn), prof->rs_insns.stddev);
        fmt_stddev(sd_ipc,  sizeof(sd_ipc),  prof->rs_ipc.stddev);
        snprintf(sd_wall, sizeof(sd_wall), "%.2f", prof->rs_wall.stddev);
        printf("cycles: %s \302\261%s  insn: %s \302\261%s"
               "  IPC: %.2f \302\261%s  wall: %.2fs \302\261%ss"
               "  (%d runs)\n",
               c_cyc, sd_cyc, c_insn, sd_insn,
               s->ipc, sd_ipc, s->wall_seconds, sd_wall, prof->n_runs);
    } else {
        printf("cycles: %s  insn: %s  IPC: %.2f  wall: %.2fs\n",
               c_cyc, c_insn, s->ipc, s->wall_seconds);
    }
    printf("cache: %.1f%% miss (%s/%s)  branch: %.1f%% miss (%s/%s)\n",
           s->cache_miss_pct, c_cm, c_cr,
           s->branch_miss_pct, c_bm, c_br);

    if (prof->has_topdown) {
        struct topdown_metrics *td = &prof->topdown;
        printf("topdown L%d: %.1f%% retiring, ", td->level, td->retiring);
        if (td->bad_spec > 0.05)
            printf("%.1f%% bad-spec, ", td->bad_spec);
        printf("%.1f%% fe-bound, %.1f%% be-bound\n",
               td->frontend, td->backend);
    }
    printf("\n");

    /* Hot functions */
    if (prof->n_funcs > 0) {
        printf("--- hot functions ---\n");
        for (int i = 0; i < prof->n_funcs; i++) {
            struct hot_func *hf = &prof->funcs[i];
            if (hf->source_file) {
                const char *sp =
                    short_path(hf->source_file, opts->source_dir);
                printf("%5.1f%%  %s;  //%s:%u-%u\n",
                       hf->overhead_pct, hf->name, sp,
                       hf->start_line, hf->end_line);
            } else {
                printf("%5.1f%%  %s;\n",
                       hf->overhead_pct, hf->name);
            }
            if (hf->skeleton_sig)
                printf("       %s\n", hf->skeleton_sig);
            if (hf->n_callees > 0) {
                printf("       -> ");
                for (int k = 0; k < hf->n_callees; k++)
                    printf("%s%s", hf->callees[k],
                           k < hf->n_callees - 1 ? ", " : "\n");
            }
            if (hf->n_callers > 0) {
                printf("       <- ");
                for (int k = 0; k < hf->n_callers; k++)
                    printf("%s (%.1f%%)%s",
                           hf->callers[k].name, hf->callers[k].pct,
                           k < hf->n_callers - 1 ? ", " : "\n");
            }
        }
        printf("\n");
    }

    /* Hot instructions (with source interleaving, sorted by address) */
    if (prof->n_insns > 0) {
        printf("--- hot instructions ---\n");
        int i = 0;
        while (i < prof->n_insns) {
            const char *cur_func = prof->insns[i].func_name;

            /* Find extent of this function group */
            int grp_start = i, grp_end = i + 1;
            while (grp_end < prof->n_insns &&
                   strcmp(prof->insns[grp_end].func_name, cur_func) == 0)
                grp_end++;
            int grp_n = grp_end - grp_start;

            /* Sort indices by address for coherent source interleaving */
            int order[grp_n]; /* VLA, N ≤ ~10 */
            for (int k = 0; k < grp_n; k++) order[k] = k;
            for (int a = 1; a < grp_n; a++) {
                int key = order[a];
                uint64_t ka = prof->insns[grp_start + key].addr;
                int j = a - 1;
                while (j >= 0 &&
                       prof->insns[grp_start + order[j]].addr > ka) {
                    order[j + 1] = order[j]; j--;
                }
                order[j + 1] = key;
            }

            /* Header: use hot_func source for the display path */
            const char *hdr_src = NULL;
            for (int f = 0; f < prof->n_funcs; f++) {
                if (strcmp(prof->funcs[f].name, cur_func) == 0 &&
                    prof->funcs[f].source_file) {
                    hdr_src = short_path(prof->funcs[f].source_file,
                                         opts->source_dir);
                    break;
                }
            }
            struct hot_insn *first = &prof->insns[grp_start + order[0]];
            if (hdr_src && first->source_line)
                printf("[%s]  //%s:%u\n",
                       cur_func, hdr_src, first->source_line);
            else
                printf("[%s]\n", cur_func);

            /* Print instructions in address order.
               Track current loaded source file; reload when it changes. */
            char *src_buf = NULL;
            char **src_lines = NULL;
            int n_src_lines = 0;
            const char *loaded_path = NULL;
            int last_printed_line = 0;

            for (int k = 0; k < grp_n; k++) {
                struct hot_insn *hi = &prof->insns[grp_start + order[k]];
                /* Switch source file if instruction maps to a different one */
                if (hi->source_file &&
                    (!loaded_path ||
                     strcmp(loaded_path, hi->source_file) != 0)) {
                    free(src_buf); src_buf = NULL;
                    free(src_lines); src_lines = NULL;
                    n_src_lines = 0;
                    last_printed_line = 0;
                    long slen;
                    src_buf = read_file_source(hi->source_file, &slen);
                    if (src_buf)
                        src_lines = index_source_lines(src_buf, slen,
                                                       &n_src_lines);
                    loaded_path = hi->source_file;
                }
                if (hi->source_line != 0 &&
                    (int)hi->source_line != last_printed_line &&
                    src_lines && (int)hi->source_line <= n_src_lines) {
                    printf("       :%u  ", hi->source_line);
                    print_source_line(src_lines[hi->source_line]);
                    printf("\n");
                    last_printed_line = (int)hi->source_line;
                }
                printf("  %5.1f%%  %s\n", hi->pct, hi->asm_text);
            }
            free(src_buf);
            free(src_lines);
            i = grp_end;
        }
        printf("\n");
    }

    /* AMDuProf */
    if (prof->n_uprof_funcs > 0) {
        printf("--- uprof (per-function) ---\n");
        printf("function,IPC,L1d miss%%,br misp/Ki,misalign/Ki\n");
        for (int i = 0; i < prof->n_uprof_funcs; i++) {
            struct uprof_func *uf = &prof->uprof_funcs[i];
            printf("%s,%.2f,%.1f,%.1f,%.1f\n",
                   uf->name, uf->ipc, uf->l1_dc_miss_pct,
                   uf->br_misp_pti, uf->misaligned_pti);
        }
        printf("\n");
    }

    /* Cache misses (with source interleaving, sorted by source line) */
    if (prof->n_cm_sites > 0) {
        printf("--- cache misses ---\n");
        int i = 0;
        while (i < prof->n_cm_sites) {
            const char *cur_func = prof->cm_sites[i].func_name;

            int grp_start = i, grp_end = i + 1;
            while (grp_end < prof->n_cm_sites &&
                   strcmp(prof->cm_sites[grp_end].func_name, cur_func) == 0)
                grp_end++;
            int grp_n = grp_end - grp_start;

            /* Sort indices by source_line */
            int order[grp_n];
            for (int k = 0; k < grp_n; k++) order[k] = k;
            for (int a = 1; a < grp_n; a++) {
                int key = order[a];
                uint32_t kl = prof->cm_sites[grp_start + key].source_line;
                if (kl == 0) kl = UINT32_MAX;
                int j = a - 1;
                while (j >= 0) {
                    uint32_t jl = prof->cm_sites[grp_start + order[j]].source_line;
                    if (jl == 0) jl = UINT32_MAX;
                    if (jl <= kl) break;
                    order[j + 1] = order[j]; j--;
                }
                order[j + 1] = key;
            }

            /* Header */
            const char *hdr_src = NULL;
            for (int f = 0; f < prof->n_funcs; f++) {
                if (strcmp(prof->funcs[f].name, cur_func) == 0 &&
                    prof->funcs[f].source_file) {
                    hdr_src = short_path(prof->funcs[f].source_file,
                                         opts->source_dir);
                    break;
                }
            }
            struct cache_miss_site *first_cm =
                &prof->cm_sites[grp_start + order[0]];
            if (hdr_src && first_cm->source_line)
                printf("[%s]  //%s:%u\n",
                       cur_func, hdr_src, first_cm->source_line);
            else
                printf("[%s]\n", cur_func);

            /* Print with per-instruction source file tracking */
            char *src_buf = NULL;
            char **src_lines = NULL;
            int n_src_lines = 0;
            const char *loaded_path = NULL;
            int last_printed_line = 0;

            for (int k = 0; k < grp_n; k++) {
                struct cache_miss_site *cm =
                    &prof->cm_sites[grp_start + order[k]];
                if (cm->source_file &&
                    (!loaded_path ||
                     strcmp(loaded_path, cm->source_file) != 0)) {
                    free(src_buf); src_buf = NULL;
                    free(src_lines); src_lines = NULL;
                    n_src_lines = 0;
                    last_printed_line = 0;
                    long slen;
                    src_buf = read_file_source(cm->source_file, &slen);
                    if (src_buf)
                        src_lines = index_source_lines(src_buf, slen,
                                                       &n_src_lines);
                    loaded_path = cm->source_file;
                }
                if (cm->source_line != 0 &&
                    (int)cm->source_line != last_printed_line &&
                    src_lines && (int)cm->source_line <= n_src_lines) {
                    printf("       :%u  ", cm->source_line);
                    print_source_line(src_lines[cm->source_line]);
                    printf("\n");
                    last_printed_line = (int)cm->source_line;
                }
                printf("  %5.1f%%  %s\n", cm->pct, cm->asm_text);
            }
            free(src_buf);
            free(src_lines);
            i = grp_end;
        }
        printf("\n");
    }

    /* Memory hotspots (cache-line level, sorted by source line) */
    if (prof->n_mem_hotspots > 0) {
        printf("--- memory hotspots ---\n");
        int i = 0;
        while (i < prof->n_mem_hotspots) {
            const char *cur_func = prof->mem_hotspots[i].func_name;

            int grp_start = i, grp_end = i + 1;
            while (grp_end < prof->n_mem_hotspots &&
                   strcmp(prof->mem_hotspots[grp_end].func_name,
                          cur_func) == 0)
                grp_end++;
            int grp_n = grp_end - grp_start;

            int order[grp_n];
            for (int k = 0; k < grp_n; k++) order[k] = k;
            for (int a = 1; a < grp_n; a++) {
                int key = order[a];
                uint32_t kl = prof->mem_hotspots[grp_start + key].source_line;
                if (kl == 0) kl = UINT32_MAX;
                int j = a - 1;
                while (j >= 0) {
                    uint32_t jl = prof->mem_hotspots[grp_start + order[j]].source_line;
                    if (jl == 0) jl = UINT32_MAX;
                    if (jl <= kl) break;
                    order[j + 1] = order[j]; j--;
                }
                order[j + 1] = key;
            }

            printf("[%s]\n", cur_func);

            /* Per-instruction source file tracking */
            char *src_buf = NULL;
            char **src_lines = NULL;
            int n_src_lines = 0;
            const char *loaded_path = NULL;
            int last_printed_line = 0;

            for (int k = 0; k < grp_n; k++) {
                struct mem_hotspot *mh =
                    &prof->mem_hotspots[grp_start + order[k]];
                if (mh->source_file &&
                    (!loaded_path ||
                     strcmp(loaded_path, mh->source_file) != 0)) {
                    free(src_buf); src_buf = NULL;
                    free(src_lines); src_lines = NULL;
                    n_src_lines = 0;
                    last_printed_line = 0;
                    long slen;
                    src_buf = read_file_source(mh->source_file, &slen);
                    if (src_buf)
                        src_lines = index_source_lines(src_buf, slen,
                                                       &n_src_lines);
                    loaded_path = mh->source_file;
                }
                if (mh->source_line != 0 &&
                    (int)mh->source_line != last_printed_line &&
                    src_lines && (int)mh->source_line <= n_src_lines) {
                    printf("       :%u  ", mh->source_line);
                    print_source_line(src_lines[mh->source_line]);
                    printf("\n");
                    last_printed_line = (int)mh->source_line;
                }
                printf("  %5.1f%%  %-40s cacheline 0x%" PRIx64
                       " (%d samples)\n",
                       mh->pct, mh->asm_text,
                       mh->cache_line << 6, mh->n_samples);
            }
            free(src_buf);
            free(src_lines);
            i = grp_end;
        }
        printf("\n");
    }

    /* llvm-mca throughput */
    if (prof->n_mca_blocks > 0) {
        printf("--- throughput (llvm-mca) ---\n");
        printf("function,RThroughput,uops,IPC,bottleneck\n");
        for (int i = 0; i < prof->n_mca_blocks; i++) {
            struct mca_block *mb = &prof->mca_blocks[i];
            printf("%s,%.2f,%d,%.2f,%s\n",
                   mb->func_name, mb->block_rthroughput,
                   mb->n_uops, mb->ipc,
                   mb->bottleneck ? mb->bottleneck : "");
        }
        printf("\n");
    }

    /* Compiler remarks */
    if (prof->n_remarks > 0) {
        printf("--- compiler remarks ---\n");
        const char *cur_func = NULL;
        for (int i = 0; i < prof->n_remarks; i++) {
            struct remark_entry *re = &prof->remarks[i];
            if (!cur_func || strcmp(cur_func, re->func_name) != 0) {
                cur_func = re->func_name;
                for (int f = 0; f < prof->n_funcs; f++) {
                    if (strcmp(prof->funcs[f].name, cur_func) == 0 &&
                        prof->funcs[f].source_file) {
                        const char *sp = short_path(
                            prof->funcs[f].source_file, opts->source_dir);
                        printf("[%s]  //%s:%u-%u\n", cur_func, sp,
                               prof->funcs[f].start_line,
                               prof->funcs[f].end_line);
                        break;
                    }
                }
            }
            printf("  :%u  %s: %s\n", re->line, re->category, re->message);
        }
        printf("\n");
    }

    /* pahole data layout */
    if (prof->n_layouts > 0) {
        printf("--- data layout ---\n");
        printf("type,bytes,cachelines,holes,padding,used_by\n");
        for (int i = 0; i < prof->n_layouts; i++) {
            struct struct_layout *sl = &prof->layouts[i];
            printf("%s,%u,%u,%u,%u,%s\n",
                   sl->type_name, sl->size, sl->cachelines,
                   sl->holes,
                   sl->padding > 0 ? sl->padding : sl->holes * 4,
                   sl->func_name);
        }
        printf("\n");
    }
}

/* ── Pipeline ────────────────────────────────────────────────────────── */

static int run_pipeline(struct perf_opts *opts, struct perf_profile *prof) {
    arena_init(&prof->arena, ARENA_DEFAULT_BLOCK);
    intern_init(&prof->strings, 256);

    int n_runs = opts->n_runs > 0 ? opts->n_runs : 1;
    prof->n_runs = n_runs;

    /* Allocate per-run stat arrays */
    prof->rs_cycles.values         = arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));
    prof->rs_insns.values          = arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));
    prof->rs_ipc.values            = arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));
    prof->rs_wall.values           = arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));
    prof->rs_cache_miss_pct.values = arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));
    prof->rs_branch_miss_pct.values= arena_alloc(&prof->arena, (size_t)n_runs * sizeof(double));

    int want_cm = (opts->cachemiss_mode >= 0);

    for (int run = 0; run < n_runs; run++) {
        fprintf(stderr, "profiling: %s (run %d/%d)\n",
                opts->cmd_str, run + 1, n_runs);

        struct raw_sample *cyc = NULL, *cm = NULL;
        int n_cyc = 0, n_cm = 0;
        uint64_t load_base = 0;
        struct arena sample_arena;
        arena_init(&sample_arena, ARENA_DEFAULT_BLOCK);

        /* Collect samples only on run 0 (instruction-level % is stable) */
        int want_samples = (run == 0);

        if (profile_child(opts, &prof->stats,
                          &prof->topdown, &prof->has_topdown,
                          want_samples ? &cyc : NULL,
                          want_samples ? &n_cyc : NULL,
                          (want_cm && want_samples) ? &cm : NULL,
                          (want_cm && want_samples) ? &n_cm : NULL,
                          &load_base, &sample_arena) != 0) {
            fprintf(stderr, "error: profiling failed (run %d)\n", run + 1);
            arena_destroy(&sample_arena);
            return -1;
        }

        /* Store per-run stats */
        prof->rs_cycles.values[run]         = (double)prof->stats.cycles;
        prof->rs_insns.values[run]          = (double)prof->stats.instructions;
        prof->rs_ipc.values[run]            = prof->stats.ipc;
        prof->rs_wall.values[run]           = prof->stats.wall_seconds;
        prof->rs_cache_miss_pct.values[run] = prof->stats.cache_miss_pct;
        prof->rs_branch_miss_pct.values[run]= prof->stats.branch_miss_pct;
        prof->rs_cycles.n = prof->rs_insns.n = prof->rs_ipc.n =
            prof->rs_wall.n = prof->rs_cache_miss_pct.n =
            prof->rs_branch_miss_pct.n = run + 1;

        /* Process samples and external tools only on run 0 */
        if (run == 0) {
            struct sym_resolver sr;
            if (symres_init(&sr, opts->binary_path, load_base) != 0) {
                fprintf(stderr,
                    "warning: symbol resolution unavailable for %s\n",
                    opts->binary_path);
            }

            int has_debug = symres_has_debug(&sr);
            if (!has_debug)
                fprintf(stderr,
                    "warning: no debug symbols in %s "
                    "(build with -g for source annotations)\n",
                    opts->binary_path);

            process_samples(&sr, cyc, n_cyc, cm, n_cm, opts, prof);

            xref_skeleton(opts, prof);
            run_mca(&sr, opts, prof);
            run_uprof(opts, prof);
            run_remarks(opts, prof);
            run_pahole(opts, prof, has_debug);

            symres_free(&sr);
        }

        arena_destroy(&sample_arena);
    }

    /* Compute mean/stddev for all metrics */
    compute_stats(&prof->rs_cycles);
    compute_stats(&prof->rs_insns);
    compute_stats(&prof->rs_ipc);
    compute_stats(&prof->rs_wall);
    compute_stats(&prof->rs_cache_miss_pct);
    compute_stats(&prof->rs_branch_miss_pct);

    /* Update stats to use mean values */
    prof->stats.cycles         = (uint64_t)prof->rs_cycles.mean;
    prof->stats.instructions   = (uint64_t)prof->rs_insns.mean;
    prof->stats.ipc            = prof->rs_ipc.mean;
    prof->stats.wall_seconds   = prof->rs_wall.mean;
    prof->stats.cache_miss_pct = prof->rs_cache_miss_pct.mean;
    prof->stats.branch_miss_pct= prof->rs_branch_miss_pct.mean;

    return 0;
}

/* ── A/B Comparison output ───────────────────────────────────────────── */

static void print_comparison(struct perf_opts *opts,
                              struct perf_profile *prof_a, const char *a_cmd,
                              struct perf_profile *prof_b, const char *b_cmd) {
    struct perf_stats *sa = &prof_a->stats;
    struct perf_stats *sb = &prof_b->stats;

    printf("=== perf A/B: %s vs %s ===\n", a_cmd, b_cmd);

    int has_stats = (prof_a->n_runs > 1 && prof_b->n_runs > 1);

    /* Stats CSV with deltas (and p-values if multi-run) */
    if (has_stats)
        printf("metric,A,B,delta,%%change,p-value,sig\n");
    else
        printf("metric,A,B,delta,%%change\n");

    {
        char ca[32], cb[32];
        fmt_count(ca, sizeof(ca), sa->cycles);
        fmt_count(cb, sizeof(cb), sb->cycles);
        char cd[32]; fmt_count(cd, sizeof(cd),
            sa->cycles > sb->cycles ? sa->cycles - sb->cycles :
                                      sb->cycles - sa->cycles);
        double pct = sa->cycles ? 100.0 * ((double)sb->cycles - sa->cycles)
                                         / sa->cycles : 0;
        if (has_stats) {
            char sda[32], sdb[32];
            fmt_stddev(sda, sizeof(sda), prof_a->rs_cycles.stddev);
            fmt_stddev(sdb, sizeof(sdb), prof_b->rs_cycles.stddev);
            double p = welch_t_test(&prof_a->rs_cycles, &prof_b->rs_cycles);
            printf("cycles,%s \302\261%s,%s \302\261%s,%s%s,%.1f%%,%.4f,%s\n",
                   ca, sda, cb, sdb,
                   sb->cycles < sa->cycles ? "-" : "+", cd, pct,
                   p, sig_marker(p));
        } else {
            printf("cycles,%s,%s,%s%s,%.1f%%\n", ca, cb,
                   sb->cycles < sa->cycles ? "-" : "+", cd, pct);
        }
    }
    {
        char ia[32], ib[32];
        fmt_count(ia, sizeof(ia), sa->instructions);
        fmt_count(ib, sizeof(ib), sb->instructions);
        char id[32]; fmt_count(id, sizeof(id),
            sa->instructions > sb->instructions ?
                sa->instructions - sb->instructions :
                sb->instructions - sa->instructions);
        double pct = sa->instructions ?
            100.0 * ((double)sb->instructions - sa->instructions)
                   / sa->instructions : 0;
        if (has_stats) {
            char sda[32], sdb[32];
            fmt_stddev(sda, sizeof(sda), prof_a->rs_insns.stddev);
            fmt_stddev(sdb, sizeof(sdb), prof_b->rs_insns.stddev);
            double p = welch_t_test(&prof_a->rs_insns, &prof_b->rs_insns);
            printf("insns,%s \302\261%s,%s \302\261%s,%s%s,%.1f%%,%.4f,%s\n",
                   ia, sda, ib, sdb,
                   sb->instructions < sa->instructions ? "-" : "+", id, pct,
                   p, sig_marker(p));
        } else {
            printf("insns,%s,%s,%s%s,%.1f%%\n", ia, ib,
                   sb->instructions < sa->instructions ? "-" : "+", id, pct);
        }
    }
    if (has_stats) {
        char sda[32], sdb[32];
        double p;
        fmt_stddev(sda, sizeof(sda), prof_a->rs_ipc.stddev);
        fmt_stddev(sdb, sizeof(sdb), prof_b->rs_ipc.stddev);
        p = welch_t_test(&prof_a->rs_ipc, &prof_b->rs_ipc);
        printf("IPC,%.2f \302\261%s,%.2f \302\261%s,%+.2f,%.1f%%,%.4f,%s\n",
               sa->ipc, sda, sb->ipc, sdb, sb->ipc - sa->ipc,
               sa->ipc ? 100.0 * (sb->ipc - sa->ipc) / sa->ipc : 0,
               p, sig_marker(p));

        snprintf(sda, sizeof(sda), "%.2f", prof_a->rs_wall.stddev);
        snprintf(sdb, sizeof(sdb), "%.2f", prof_b->rs_wall.stddev);
        p = welch_t_test(&prof_a->rs_wall, &prof_b->rs_wall);
        printf("wall,%.2fs \302\261%ss,%.2fs \302\261%ss,%+.2fs,%.1f%%,%.4f,%s\n",
               sa->wall_seconds, sda, sb->wall_seconds, sdb,
               sb->wall_seconds - sa->wall_seconds,
               sa->wall_seconds ? 100.0 * (sb->wall_seconds - sa->wall_seconds)
                                  / sa->wall_seconds : 0,
               p, sig_marker(p));

        p = welch_t_test(&prof_a->rs_cache_miss_pct, &prof_b->rs_cache_miss_pct);
        printf("cache miss%%,%.1f%%,%.1f%%,%+.1fpp,%.1f%%,%.4f,%s\n",
               sa->cache_miss_pct, sb->cache_miss_pct,
               sb->cache_miss_pct - sa->cache_miss_pct,
               sa->cache_miss_pct ? 100.0 * (sb->cache_miss_pct - sa->cache_miss_pct)
                                    / sa->cache_miss_pct : 0,
               p, sig_marker(p));

        p = welch_t_test(&prof_a->rs_branch_miss_pct, &prof_b->rs_branch_miss_pct);
        printf("branch miss%%,%.1f%%,%.1f%%,%+.1fpp,%.1f%%,%.4f,%s\n",
               sa->branch_miss_pct, sb->branch_miss_pct,
               sb->branch_miss_pct - sa->branch_miss_pct,
               sa->branch_miss_pct ? 100.0 * (sb->branch_miss_pct - sa->branch_miss_pct)
                                      / sa->branch_miss_pct : 0,
               p, sig_marker(p));
    } else {
        printf("IPC,%.2f,%.2f,%+.2f,%.1f%%\n",
               sa->ipc, sb->ipc, sb->ipc - sa->ipc,
               sa->ipc ? 100.0 * (sb->ipc - sa->ipc) / sa->ipc : 0);
        printf("wall,%.2fs,%.2fs,%+.2fs,%.1f%%\n",
               sa->wall_seconds, sb->wall_seconds,
               sb->wall_seconds - sa->wall_seconds,
               sa->wall_seconds ? 100.0 * (sb->wall_seconds - sa->wall_seconds)
                                  / sa->wall_seconds : 0);
        printf("cache miss%%,%.1f%%,%.1f%%,%+.1fpp,%.1f%%\n",
               sa->cache_miss_pct, sb->cache_miss_pct,
               sb->cache_miss_pct - sa->cache_miss_pct,
               sa->cache_miss_pct ? 100.0 * (sb->cache_miss_pct - sa->cache_miss_pct)
                                    / sa->cache_miss_pct : 0);
        printf("branch miss%%,%.1f%%,%.1f%%,%+.1fpp,%.1f%%\n",
               sa->branch_miss_pct, sb->branch_miss_pct,
               sb->branch_miss_pct - sa->branch_miss_pct,
               sa->branch_miss_pct ? 100.0 * (sb->branch_miss_pct - sa->branch_miss_pct)
                                      / sa->branch_miss_pct : 0);
    }
    printf("\n");

    /* Hot function delta */
    printf("--- hot functions (A -> B) ---\n");
    printf("  A%%    B%%   delta  function\n");

    for (int b = 0; b < prof_b->n_funcs; b++) {
        char bclean[256];
        snprintf(bclean, sizeof(bclean), "%s", prof_b->funcs[b].name);
        strip_compiler_suffix(bclean);

        double a_pct = 0;
        int a_idx = -1;
        for (int a = 0; a < prof_a->n_funcs; a++) {
            char aclean[256];
            snprintf(aclean, sizeof(aclean), "%s", prof_a->funcs[a].name);
            strip_compiler_suffix(aclean);
            if (strcmp(aclean, bclean) == 0) {
                a_pct = prof_a->funcs[a].overhead_pct;
                a_idx = a;
                break;
            }
        }

        double delta = prof_b->funcs[b].overhead_pct - a_pct;
        const char *src = NULL;
        if (prof_b->funcs[b].source_file)
            src = short_path(prof_b->funcs[b].source_file, opts->source_dir);
        printf("%5.1f  %5.1f  %+5.1f   %s",
               a_pct, prof_b->funcs[b].overhead_pct, delta,
               prof_b->funcs[b].name);
        if (a_idx < 0) printf("  [NEW]");
        if (src) printf("  //%s", src);
        printf("\n");

        /* Caller delta */
        struct hot_func *hfb = &prof_b->funcs[b];
        struct hot_func *hfa = a_idx >= 0 ? &prof_a->funcs[a_idx] : NULL;
        if (hfb->n_callers > 0 || (hfa && hfa->n_callers > 0)) {
            printf("       <- ");
            int first = 1;
            for (int c = 0; c < hfb->n_callers; c++) {
                if (!first) printf(", ");
                first = 0;
                char cclean[256];
                snprintf(cclean, sizeof(cclean), "%s", hfb->callers[c].name);
                strip_compiler_suffix(cclean);
                double a_cpct = -1;
                if (hfa) {
                    for (int ac = 0; ac < hfa->n_callers; ac++) {
                        char acclean[256];
                        snprintf(acclean, sizeof(acclean), "%s",
                                 hfa->callers[ac].name);
                        strip_compiler_suffix(acclean);
                        if (strcmp(cclean, acclean) == 0) {
                            a_cpct = hfa->callers[ac].pct;
                            break;
                        }
                    }
                }
                printf("%s (", hfb->callers[c].name);
                if (a_cpct >= 0)
                    printf("%.0f%% -> %.0f%%", a_cpct, hfb->callers[c].pct);
                else
                    printf("%.0f%% [NEW]", hfb->callers[c].pct);
                printf(")");
            }
            /* A-only callers (gone in B) */
            if (hfa) {
                for (int ac = 0; ac < hfa->n_callers; ac++) {
                    char acclean[256];
                    snprintf(acclean, sizeof(acclean), "%s",
                             hfa->callers[ac].name);
                    strip_compiler_suffix(acclean);
                    int in_b = 0;
                    for (int c = 0; c < hfb->n_callers; c++) {
                        char bcclean[256];
                        snprintf(bcclean, sizeof(bcclean), "%s",
                                 hfb->callers[c].name);
                        strip_compiler_suffix(bcclean);
                        if (strcmp(acclean, bcclean) == 0) { in_b = 1; break; }
                    }
                    if (!in_b) {
                        if (!first) printf(", ");
                        first = 0;
                        printf("%s (%.0f%% [GONE])",
                               hfa->callers[ac].name, hfa->callers[ac].pct);
                    }
                }
            }
            printf("\n");
        }
    }

    /* Print A-only functions (GONE in B) */
    for (int a = 0; a < prof_a->n_funcs; a++) {
        char aclean[256];
        snprintf(aclean, sizeof(aclean), "%s", prof_a->funcs[a].name);
        strip_compiler_suffix(aclean);

        int found_b = 0;
        for (int b = 0; b < prof_b->n_funcs; b++) {
            char bclean[256];
            snprintf(bclean, sizeof(bclean), "%s", prof_b->funcs[b].name);
            strip_compiler_suffix(bclean);
            if (strcmp(aclean, bclean) == 0) { found_b = 1; break; }
        }
        if (!found_b) {
            printf("%5.1f   0.0  %+5.1f   %s  [GONE]\n",
                   prof_a->funcs[a].overhead_pct,
                   -prof_a->funcs[a].overhead_pct,
                   prof_a->funcs[a].name);
        }
    }
    printf("\n");

    /* Per-function cache miss hotspots (sorted by source line) */
    if (prof_a->n_cm_sites > 0 && prof_b->n_cm_sites > 0) {
        printf("--- cache miss hotspots A -> B ---\n");
        for (int b = 0; b < prof_b->n_funcs; b++) {
            char bclean[256];
            snprintf(bclean, sizeof(bclean), "%s", prof_b->funcs[b].name);
            strip_compiler_suffix(bclean);

            int found_a = 0;
            for (int a = 0; a < prof_a->n_funcs; a++) {
                char aclean[256];
                snprintf(aclean, sizeof(aclean), "%s",
                         prof_a->funcs[a].name);
                strip_compiler_suffix(aclean);
                if (strcmp(aclean, bclean) == 0) { found_a = 1; break; }
            }
            if (!found_a) continue;

            printf("[%s]\n", prof_b->funcs[b].name);

            /* Collect top 3 for each side, sort by source_line */
            struct { int idx; char side; } entries[6];
            int n_entries = 0;
            int count = 0;
            for (int i = 0; i < prof_a->n_cm_sites && count < 3; i++) {
                char cmn[256];
                snprintf(cmn, sizeof(cmn), "%s", prof_a->cm_sites[i].func_name);
                strip_compiler_suffix(cmn);
                if (strcmp(cmn, bclean) == 0)
                    entries[n_entries++] = (typeof(entries[0])){i, 'A'}, count++;
            }
            count = 0;
            for (int i = 0; i < prof_b->n_cm_sites && count < 3; i++) {
                char cmn[256];
                snprintf(cmn, sizeof(cmn), "%s", prof_b->cm_sites[i].func_name);
                strip_compiler_suffix(cmn);
                if (strcmp(cmn, bclean) == 0)
                    entries[n_entries++] = (typeof(entries[0])){i, 'B'}, count++;
            }
            /* Sort by source_line */
            for (int a = 1; a < n_entries; a++) {
                typeof(entries[0]) key = entries[a];
                uint32_t kl = key.side == 'A'
                    ? prof_a->cm_sites[key.idx].source_line
                    : prof_b->cm_sites[key.idx].source_line;
                if (kl == 0) kl = UINT32_MAX;
                int j = a - 1;
                while (j >= 0) {
                    uint32_t jl = entries[j].side == 'A'
                        ? prof_a->cm_sites[entries[j].idx].source_line
                        : prof_b->cm_sites[entries[j].idx].source_line;
                    if (jl == 0) jl = UINT32_MAX;
                    if (jl <= kl) break;
                    entries[j + 1] = entries[j]; j--;
                }
                entries[j + 1] = key;
            }

            /* Per-instruction source file tracking */
            char *sbuf = NULL; char **slines = NULL; int nslines = 0;
            const char *loaded_path = NULL;
            int last_line = 0;
            for (int k = 0; k < n_entries; k++) {
                uint32_t sl; const char *sf;
                if (entries[k].side == 'A') {
                    sl = prof_a->cm_sites[entries[k].idx].source_line;
                    sf = prof_a->cm_sites[entries[k].idx].source_file;
                } else {
                    sl = prof_b->cm_sites[entries[k].idx].source_line;
                    sf = prof_b->cm_sites[entries[k].idx].source_file;
                }
                if (sf && (!loaded_path ||
                           strcmp(loaded_path, sf) != 0)) {
                    free(sbuf); sbuf = NULL;
                    free(slines); slines = NULL;
                    nslines = 0; last_line = 0;
                    long slen;
                    sbuf = read_file_source(sf, &slen);
                    if (sbuf)
                        slines = index_source_lines(sbuf, slen, &nslines);
                    loaded_path = sf;
                }
                if (sl != 0 && (int)sl != last_line &&
                    slines && (int)sl <= nslines) {
                    printf("       :%u  ", sl);
                    print_source_line(slines[sl]);
                    printf("\n");
                    last_line = (int)sl;
                }
                if (entries[k].side == 'A')
                    printf("  A: %5.1f%%  %s\n",
                           prof_a->cm_sites[entries[k].idx].pct,
                           prof_a->cm_sites[entries[k].idx].asm_text);
                else
                    printf("  B: %5.1f%%  %s\n",
                           prof_b->cm_sites[entries[k].idx].pct,
                           prof_b->cm_sites[entries[k].idx].asm_text);
            }
            free(sbuf); free(slines);
        }
        printf("\n");
    }

    /* Hot instruction summary (sorted by source line) */
    if (prof_a->n_insns > 0 && prof_b->n_insns > 0) {
        printf("--- hot instructions A -> B ---\n");
        for (int b = 0; b < prof_b->n_funcs; b++) {
            char bclean[256];
            snprintf(bclean, sizeof(bclean), "%s", prof_b->funcs[b].name);
            strip_compiler_suffix(bclean);

            int found_a = 0;
            for (int a = 0; a < prof_a->n_funcs; a++) {
                char aclean[256];
                snprintf(aclean, sizeof(aclean), "%s",
                         prof_a->funcs[a].name);
                strip_compiler_suffix(aclean);
                if (strcmp(aclean, bclean) == 0) { found_a = 1; break; }
            }
            if (!found_a) continue;

            printf("[%s]\n", prof_b->funcs[b].name);

            /* Collect top 3 for each side, sort by source_line */
            struct { int idx; char side; } entries[6];
            int n_entries = 0;
            int count = 0;
            for (int i = 0; i < prof_a->n_insns && count < 3; i++) {
                char iclean[256];
                snprintf(iclean, sizeof(iclean), "%s", prof_a->insns[i].func_name);
                strip_compiler_suffix(iclean);
                if (strcmp(iclean, bclean) == 0)
                    entries[n_entries++] = (typeof(entries[0])){i, 'A'}, count++;
            }
            count = 0;
            for (int i = 0; i < prof_b->n_insns && count < 3; i++) {
                char iclean[256];
                snprintf(iclean, sizeof(iclean), "%s", prof_b->insns[i].func_name);
                strip_compiler_suffix(iclean);
                if (strcmp(iclean, bclean) == 0)
                    entries[n_entries++] = (typeof(entries[0])){i, 'B'}, count++;
            }
            /* Sort by source_line */
            for (int a = 1; a < n_entries; a++) {
                typeof(entries[0]) key = entries[a];
                uint32_t kl = key.side == 'A'
                    ? prof_a->insns[key.idx].source_line
                    : prof_b->insns[key.idx].source_line;
                if (kl == 0) kl = UINT32_MAX;
                int j = a - 1;
                while (j >= 0) {
                    uint32_t jl = entries[j].side == 'A'
                        ? prof_a->insns[entries[j].idx].source_line
                        : prof_b->insns[entries[j].idx].source_line;
                    if (jl == 0) jl = UINT32_MAX;
                    if (jl <= kl) break;
                    entries[j + 1] = entries[j]; j--;
                }
                entries[j + 1] = key;
            }

            /* Per-instruction source file tracking */
            char *sbuf = NULL; char **slines = NULL; int nslines = 0;
            const char *loaded_path = NULL;
            int last_line = 0;
            for (int k = 0; k < n_entries; k++) {
                uint32_t sl; const char *sf;
                if (entries[k].side == 'A') {
                    sl = prof_a->insns[entries[k].idx].source_line;
                    sf = prof_a->insns[entries[k].idx].source_file;
                } else {
                    sl = prof_b->insns[entries[k].idx].source_line;
                    sf = prof_b->insns[entries[k].idx].source_file;
                }
                if (sf && (!loaded_path ||
                           strcmp(loaded_path, sf) != 0)) {
                    free(sbuf); sbuf = NULL;
                    free(slines); slines = NULL;
                    nslines = 0; last_line = 0;
                    long slen;
                    sbuf = read_file_source(sf, &slen);
                    if (sbuf)
                        slines = index_source_lines(sbuf, slen, &nslines);
                    loaded_path = sf;
                }
                if (sl != 0 && (int)sl != last_line &&
                    slines && (int)sl <= nslines) {
                    printf("       :%u  ", sl);
                    print_source_line(slines[sl]);
                    printf("\n");
                    last_line = (int)sl;
                }
                if (entries[k].side == 'A')
                    printf("  A: %5.1f%%  %s\n",
                           prof_a->insns[entries[k].idx].pct,
                           prof_a->insns[entries[k].idx].asm_text);
                else
                    printf("  B: %5.1f%%  %s\n",
                           prof_b->insns[entries[k].idx].pct,
                           prof_b->insns[entries[k].idx].asm_text);
            }
            free(sbuf); free(slines);
        }
        printf("\n");
    }

    /* Memory hotspots A -> B */
    if (prof_a->n_mem_hotspots > 0 || prof_b->n_mem_hotspots > 0) {
        printf("--- memory hotspots A -> B ---\n");
        for (int b = 0; b < prof_b->n_funcs; b++) {
            char bclean[256];
            snprintf(bclean, sizeof(bclean), "%s", prof_b->funcs[b].name);
            strip_compiler_suffix(bclean);

            /* Check if this function has memory hotspots in either profile */
            int has_a = 0, has_b = 0;
            for (int i = 0; i < prof_a->n_mem_hotspots && !has_a; i++)
                if (strcmp(prof_a->mem_hotspots[i].func_name, bclean) == 0)
                    has_a = 1;
            for (int i = 0; i < prof_b->n_mem_hotspots && !has_b; i++)
                if (strcmp(prof_b->mem_hotspots[i].func_name, bclean) == 0)
                    has_b = 1;
            if (!has_a && !has_b) continue;

            printf("[%s]\n", prof_b->funcs[b].name);
            int count = 0;
            for (int i = 0; i < prof_a->n_mem_hotspots && count < 3; i++) {
                if (strcmp(prof_a->mem_hotspots[i].func_name, bclean) == 0) {
                    printf("  A: %5.1f%%  %-40s cacheline 0x%" PRIx64 "\n",
                           prof_a->mem_hotspots[i].pct,
                           prof_a->mem_hotspots[i].asm_text,
                           prof_a->mem_hotspots[i].cache_line << 6);
                    count++;
                }
            }
            count = 0;
            for (int i = 0; i < prof_b->n_mem_hotspots && count < 3; i++) {
                if (strcmp(prof_b->mem_hotspots[i].func_name, bclean) == 0) {
                    printf("  B: %5.1f%%  %-40s cacheline 0x%" PRIx64 "\n",
                           prof_b->mem_hotspots[i].pct,
                           prof_b->mem_hotspots[i].asm_text,
                           prof_b->mem_hotspots[i].cache_line << 6);
                    count++;
                }
            }
        }
        printf("\n");
    }

    /* MCA throughput delta */
    if (prof_a->n_mca_blocks > 0 || prof_b->n_mca_blocks > 0) {
        printf("--- throughput A -> B (llvm-mca) ---\n");
        printf("function,A_rthru,B_rthru,delta,A_IPC,B_IPC\n");
        for (int b = 0; b < prof_b->n_mca_blocks; b++) {
            char bclean[256];
            snprintf(bclean, sizeof(bclean), "%s",
                     prof_b->mca_blocks[b].func_name);
            strip_compiler_suffix(bclean);

            double a_rt = 0, a_ipc = 0;
            for (int a = 0; a < prof_a->n_mca_blocks; a++) {
                char aclean[256];
                snprintf(aclean, sizeof(aclean), "%s",
                         prof_a->mca_blocks[a].func_name);
                strip_compiler_suffix(aclean);
                if (strcmp(aclean, bclean) == 0) {
                    a_rt = prof_a->mca_blocks[a].block_rthroughput;
                    a_ipc = prof_a->mca_blocks[a].ipc;
                    break;
                }
            }
            printf("%s,%.2f,%.2f,%+.2f,%.2f,%.2f\n",
                   prof_b->mca_blocks[b].func_name,
                   a_rt, prof_b->mca_blocks[b].block_rthroughput,
                   prof_b->mca_blocks[b].block_rthroughput - a_rt,
                   a_ipc, prof_b->mca_blocks[b].ipc);
        }
        printf("\n");
    }

    /* Topdown delta */
    if (prof_a->has_topdown && prof_b->has_topdown) {
        struct topdown_metrics *ta = &prof_a->topdown;
        struct topdown_metrics *tb = &prof_b->topdown;
        printf("--- topdown A -> B ---\n");
        printf("category,A,B,delta\n");
        printf("retiring,%.1f%%,%.1f%%,%+.1fpp\n",
               ta->retiring, tb->retiring, tb->retiring - ta->retiring);
        if (ta->bad_spec > 0.05 || tb->bad_spec > 0.05)
            printf("bad_spec,%.1f%%,%.1f%%,%+.1fpp\n",
                   ta->bad_spec, tb->bad_spec, tb->bad_spec - ta->bad_spec);
        printf("frontend,%.1f%%,%.1f%%,%+.1fpp\n",
               ta->frontend, tb->frontend, tb->frontend - ta->frontend);
        printf("backend,%.1f%%,%.1f%%,%+.1fpp\n",
               ta->backend, tb->backend, tb->backend - ta->backend);
        printf("\n");
    }

    /* Remarks diff */
    if (prof_a->n_remarks > 0 || prof_b->n_remarks > 0) {
        printf("--- remarks diff (A -> B) ---\n");
        int *b_matched = xcalloc((size_t)prof_b->n_remarks, sizeof(int));
        int any_diff = 0;
        const char *cur_func = NULL;

        for (int i = 0; i < prof_a->n_remarks; i++) {
            struct remark_entry *ra = &prof_a->remarks[i];
            int found = 0;
            for (int j = 0; j < prof_b->n_remarks; j++) {
                struct remark_entry *rb = &prof_b->remarks[j];
                if (strcmp(ra->category, rb->category) == 0 &&
                    strcmp(ra->message, rb->message) == 0 &&
                    strcmp(ra->func_name, rb->func_name) == 0) {
                    b_matched[j] = 1;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (!cur_func || strcmp(cur_func, ra->func_name) != 0) {
                    cur_func = ra->func_name;
                    printf("[%s]\n", cur_func);
                }
                printf("  A only: :%u  %s: %s\n",
                       ra->line, ra->category, ra->message);
                any_diff = 1;
            }
        }
        cur_func = NULL;
        for (int j = 0; j < prof_b->n_remarks; j++) {
            if (b_matched[j]) continue;
            struct remark_entry *rb = &prof_b->remarks[j];
            if (!cur_func || strcmp(cur_func, rb->func_name) != 0) {
                cur_func = rb->func_name;
                printf("[%s]\n", cur_func);
            }
            printf("  B only: :%u  %s: %s\n",
                   rb->line, rb->category, rb->message);
            any_diff = 1;
        }
        free(b_matched);
        if (!any_diff)
            printf("  (no differences)\n");
        printf("\n");
    }
}

/* ── Cleanup ─────────────────────────────────────────────────────────── */

static void free_profile(struct perf_profile *prof) {
    intern_destroy(&prof->strings);
    arena_destroy(&prof->arena);
}

/* ── Entry point ─────────────────────────────────────────────────────── */

int perf_main(int argc, char *argv[]) {
    struct perf_opts opts = {
        .top_n      = 10,
        .insns_n    = 10,
        .build_cmd  = "make",
        .no_build   = 0,
        .uprof_mode = 0,
        .source_dir = ".",
        .verbose    = 0,
    };

    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "n:i:b:s:vh",
                              perf_long_options, NULL)) != -1) {
        switch (opt) {
        case 'n': opts.top_n    = atoi(optarg); break;
        case 'i': opts.insns_n  = atoi(optarg); break;
        case 'b': opts.build_cmd = optarg;      break;
        case PERF_OPT_NO_BUILD: opts.no_build   = 1;  break;
        case PERF_OPT_UPROF:       opts.uprof_mode    =  1; break;
        case PERF_OPT_NO_UPROF:    opts.uprof_mode    = -1; break;
        case PERF_OPT_TOPDOWN:     opts.topdown_mode  =  1; break;
        case PERF_OPT_NO_TOPDOWN:  opts.topdown_mode  = -1; break;
        case PERF_OPT_MCA:         opts.mca_mode      =  1; break;
        case PERF_OPT_NO_MCA:      opts.mca_mode      = -1; break;
        case PERF_OPT_CACHEMISS:   opts.cachemiss_mode =  1; break;
        case PERF_OPT_NO_CACHEMISS:opts.cachemiss_mode = -1; break;
        case PERF_OPT_PAHOLE:      opts.pahole_mode   =  1; break;
        case PERF_OPT_NO_PAHOLE:   opts.pahole_mode   = -1; break;
        case PERF_OPT_VS:          opts.vs_binary     = optarg; break;
        case PERF_OPT_REMARKS:     opts.remarks_mode  =  1; break;
        case PERF_OPT_NO_REMARKS:  opts.remarks_mode  = -1; break;
        case PERF_OPT_RUNS:        opts.n_runs        = atoi(optarg); break;
        case 's': opts.source_dir = optarg;     break;
        case 'v': opts.verbose    = 1;          break;
        case 'h': perf_usage(argv[0]); return 0;
        default:  perf_usage(argv[0]); return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "error: no command specified after --\n");
        perf_usage(argv[0]);
        return 1;
    }

    opts.cmd_argv = argv + optind;
    opts.cmd_argc = argc - optind;
    opts.cmd_str = join_argv(opts.cmd_argv, opts.cmd_argc);
    opts.binary_path = resolve_binary(opts.cmd_argv[0]);

    /* Temp directory for llvm-mca and uprof temp files */
    snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/archmap-XXXXXX");
    if (!mkdtemp(g_tmpdir)) {
        perror("mkdtemp");
        free(opts.cmd_str);
        free(opts.binary_path);
        return 1;
    }
    atexit(cleanup_tmpdir);

    struct perf_profile prof;
    memset(&prof, 0, sizeof(prof));

    /* Phase 1: Build */
    if (phase_build(&opts) != 0) goto fail;

    /* Check prerequisites */
    if (check_perf_access() != 0) goto fail;

    if (opts.vs_binary) {
        /* ── A/B comparison mode ─────────────────────────────── */
        /* Default to 5 runs for A/B if not explicitly set */
        if (opts.n_runs == 0) opts.n_runs = 5;

        char top_tmpdir[PATH_MAX];
        strcpy(top_tmpdir, g_tmpdir);

        /* --- Run A (baseline) --- */
        char *a_binary = resolve_binary(opts.vs_binary);
        if (access(a_binary, X_OK) != 0) {
            fprintf(stderr, "error: baseline binary not executable: %s\n",
                    a_binary);
            free(a_binary);
            goto fail;
        }

        char **a_argv = xmalloc((size_t)opts.cmd_argc * sizeof(char *));
        a_argv[0] = a_binary;
        for (int i = 1; i < opts.cmd_argc; i++)
            a_argv[i] = opts.cmd_argv[i];
        char *a_cmd_str = join_argv(a_argv, opts.cmd_argc);

        char *b_cmd_str = opts.cmd_str;
        char *b_binary = opts.binary_path;
        char **b_argv = opts.cmd_argv;

        /* Run A pipeline */
        char a_dir[PATH_MAX];
        snprintf(a_dir, sizeof(a_dir), "%s/a", top_tmpdir);
        mkdir(a_dir, 0700);
        strcpy(g_tmpdir, a_dir);

        opts.cmd_str = a_cmd_str;
        opts.binary_path = a_binary;
        opts.cmd_argv = a_argv;
        int saved_no_build = opts.no_build;
        opts.no_build = 1;

        struct perf_profile prof_a;
        memset(&prof_a, 0, sizeof(prof_a));

        fprintf(stderr, "=== A (baseline): %s ===\n", a_cmd_str);
        int a_ok = run_pipeline(&opts, &prof_a);

        /* Run B pipeline */
        char b_dir[PATH_MAX];
        snprintf(b_dir, sizeof(b_dir), "%s/b", top_tmpdir);
        mkdir(b_dir, 0700);
        strcpy(g_tmpdir, b_dir);

        opts.cmd_str = b_cmd_str;
        opts.binary_path = b_binary;
        opts.cmd_argv = b_argv;
        opts.no_build = saved_no_build;

        struct perf_profile prof_b;
        memset(&prof_b, 0, sizeof(prof_b));

        fprintf(stderr, "=== B (new): %s ===\n", b_cmd_str);
        int b_ok = run_pipeline(&opts, &prof_b);

        /* Restore g_tmpdir for cleanup */
        strcpy(g_tmpdir, top_tmpdir);

        if (a_ok == 0 && b_ok == 0)
            print_comparison(&opts, &prof_a, a_cmd_str,
                             &prof_b, b_cmd_str);
        else
            fprintf(stderr, "error: pipeline failed "
                    "(A=%s, B=%s)\n",
                    a_ok ? "FAIL" : "ok", b_ok ? "FAIL" : "ok");

        free_profile(&prof_a);
        free_profile(&prof_b);
        free(a_cmd_str);
        free(a_argv);
        free(a_binary);
        free(opts.cmd_str);
        free(opts.binary_path);
        return (a_ok == 0 && b_ok == 0) ? 0 : 1;
    }

    /* ── Normal single-profile path ──────────────────────── */
    if (run_pipeline(&opts, &prof) != 0)
        goto fail;

    print_report(&opts, &prof);

    free_profile(&prof);
    free(opts.cmd_str);
    free(opts.binary_path);
    return 0;

fail:
    free_profile(&prof);
    free(opts.cmd_str);
    free(opts.binary_path);
    return 1;
}
