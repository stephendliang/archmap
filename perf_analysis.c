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
#include "arena.h"
#include "symres.h"
#include "perf_tools.h"
#include "perf_report.h"

#ifndef PERF_FLAG_FD_CLOEXEC
#define PERF_FLAG_FD_CLOEXEC (1UL << 3)
#endif

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

char g_tmpdir[PATH_MAX];

static void cleanup_tmpdir(void) {
    if (g_tmpdir[0] == '\0') return;
    char cmd[PATH_MAX + 16];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", g_tmpdir);
    system(cmd);
    g_tmpdir[0] = '\0';
}

char *run_cmd(const char *cmd, int *out_status, int verbose) {
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

void fmt_count(char *buf, size_t sz, uint64_t val) {
    if (val >= 1000000000ULL)
        snprintf(buf, sz, "%.2fG", val / 1e9);
    else if (val >= 1000000ULL)
        snprintf(buf, sz, "%.1fM", val / 1e6);
    else if (val >= 1000ULL)
        snprintf(buf, sz, "%.0fK", val / 1e3);
    else
        snprintf(buf, sz, "%" PRIu64, val);
}

void strip_compiler_suffix(char *name) {
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

int has_tool(const char *name) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "which '%s' >/dev/null 2>&1", name);
    return system(cmd) == 0;
}

/* Resolve a tool to its full path, checking PATH then well-known dirs.
   Returns static buffer — valid until next call. */
const char *find_tool(const char *name) {
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
int is_boring_caller(const char *name) {
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

void add_caller_entry(struct perf_profile *prof, struct hot_func *hf,
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
