/* perf_analysis.c — Performance analysis pipeline for archmap
 *
 * Pipeline: build → profile (perf stat/record/annotate, AMDuProf) →
 *           skeleton cross-reference → compact report
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

#include <tree_sitter/api.h>
#include "perf_analysis.h"
#include "git_cache.h"

/* ── Options ─────────────────────────────────────────────────────────── */

struct perf_opts {
    int top_n;
    int insns_n;
    int runs;
    const char *build_cmd;
    int no_build;
    int uprof_mode;       /* 1=force, -1=skip, 0=auto */
    int topdown_mode;     /* 0=auto, 1=force, -1=skip */
    int mca_mode;
    int cachemiss_mode;
    int pahole_mode;
    int remarks_mode;     /* 0=auto, 1=force, -1=skip */
    const char *source_dir;
    int verbose;
    char **cmd_argv;
    int cmd_argc;
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
};

static struct option perf_long_options[] = {
    {"top",       required_argument, NULL, 'n'},
    {"insns",     required_argument, NULL, 'i'},
    {"runs",      required_argument, NULL, 'r'},
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
        "  -r, --runs N         perf stat repetitions (default 3)\n"
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

static char *run_cmd(const char *cmd, int *out_status, int verbose) {
    if (verbose) fprintf(stderr, "+ %s\n", cmd);
    FILE *fp = popen(cmd, "r");
    if (!fp) { if (out_status) *out_status = -1; return NULL; }

    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, fp)) > 0) {
        len += n;
        if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
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
    char *cmd = malloc(total + 1);
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
    if (name[0] == '/' || name[0] == '.') {
        char *real = realpath(name, NULL);
        if (real) return real;
    }
    char cmd[PATH_MAX + 32];
    snprintf(cmd, sizeof(cmd), "which '%s' 2>/dev/null", name);
    int status;
    char *out = run_cmd(cmd, &status, 0);
    if (status == 0 && out && out[0]) {
        size_t len = strlen(out);
        while (len > 0 && (out[len-1] == '\n' || out[len-1] == '\r'))
            out[--len] = '\0';
        return out;
    }
    free(out);
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

static int check_debug_symbols(const char *binary) {
    char cmd[PATH_MAX + 64];
    snprintf(cmd, sizeof(cmd),
        "readelf -S '%s' 2>/dev/null", binary);
    int status;
    char *out = run_cmd(cmd, &status, 0);
    int has_debug = (out && strstr(out, ".debug_info") != NULL);
    free(out);
    return has_debug;
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

/* ── Phase 2a: perf stat ─────────────────────────────────────────────── */

static int run_perf_stat(struct perf_opts *opts, struct perf_stats *stats) {
    memset(stats, 0, sizeof(*stats));

    char cmd[4096];
    /* 2>&1 1>/dev/null: capture perf's stderr (CSV data) via pipe,
       suppress profiled command's stdout */
    snprintf(cmd, sizeof(cmd),
        "perf stat -e duration_time,cycles,instructions,"
        "cache-references,cache-misses,branches,branch-misses "
        "-x ',' -r %d -- %s 2>&1 1>/dev/null",
        opts->runs, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    if (!out) { fprintf(stderr, "perf stat failed to launch\n"); return -1; }

    /* Parse CSV lines: value,,event_name,time_running,pct,, */
    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        uint64_t val;
        char event[128];
        if (sscanf(line, "%" SCNu64 ",,%127[^,]", &val, event) >= 2) {
            if (strstr(event, "duration_time"))
                stats->wall_seconds = val / 1e9;
            else if (strstr(event, "branch-misses"))
                stats->branch_misses = val;
            else if (strstr(event, "cache-misses"))
                stats->cache_misses = val;
            else if (strstr(event, "cache-references"))
                stats->cache_refs = val;
            else if (strstr(event, "instructions"))
                stats->instructions = val;
            else if (strstr(event, "cycles") &&
                     !strstr(event, "stalled"))
                stats->cycles = val;
            else if (strstr(event, "branches") &&
                     !strstr(event, "branch-misses"))
                stats->branches = val;
        }

        line = nl ? nl + 1 : NULL;
    }
    free(out);

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

/* ── Phase 2b: perf record + report ──────────────────────────────────── */

static int run_perf_record(struct perf_opts *opts) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "perf record -g --call-graph dwarf "
        "-o '%s/perf.data' -- %s 2>&1 1>/dev/null",
        g_tmpdir, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    free(out);

    /* perf record passes through the profiled command's exit code.
       Check if perf.data was actually created instead. */
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/perf.data", g_tmpdir);
    if (access(path, F_OK) != 0) {
        fprintf(stderr, "perf record failed — no perf.data created\n");
        return -1;
    }
    return 0;
}

static int parse_perf_report(struct perf_opts *opts, struct perf_profile *prof) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "perf report --stdio --no-children --percent-limit 0.5 "
        "-i '%s/perf.data' 2>/dev/null",
        g_tmpdir);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    if (!out) return -1;

    int cap = 32;
    prof->funcs = malloc((size_t)cap * sizeof(*prof->funcs));
    prof->n_funcs = 0;

    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        /* Match: " 45.23%  cmd  obj  [.] func_name" */
        char *bracket = strstr(line, "[.] ");
        if (bracket) {
            double pct;
            if (sscanf(line, " %lf%%", &pct) == 1) {
                char *name = bracket + 4;
                size_t nlen = strlen(name);
                while (nlen > 0 &&
                       (name[nlen-1] == ' ' || name[nlen-1] == '\n'))
                    nlen--;

                if (nlen > 0 && prof->n_funcs < opts->top_n) {
                    if (prof->n_funcs >= cap) {
                        cap *= 2;
                        prof->funcs = realloc(prof->funcs,
                            (size_t)cap * sizeof(*prof->funcs));
                    }
                    struct hot_func *hf =
                        &prof->funcs[prof->n_funcs++];
                    memset(hf, 0, sizeof(*hf));
                    hf->name = strndup(name, nlen);
                    hf->overhead_pct = pct;
                }
            }
        }

        line = nl ? nl + 1 : NULL;
    }
    free(out);
    return 0;
}

/* ── Phase 2b2: caller context ───────────────────────────────────────── */

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

/* Extract a clean function name from a tree line.
   Returns pointer into 'buf' or NULL.  Strips " (inlined)" suffix. */
static char *extract_tree_name(const char *raw, char *buf, size_t bufsz) {
    const char *p = raw;
    while (*p == ' ' || *p == '\t' || *p == '|') p++;
    if (!*p || *p == '#' || *p == '\n') return NULL;
    size_t len = strlen(p);
    while (len > 0 && (p[len-1] == ' ' || p[len-1] == '\n')) len--;
    /* Strip " (inlined)" suffix */
    if (len > 10 && strncmp(p + len - 10, " (inlined)", 10) == 0)
        len -= 10;
    if (len == 0 || len >= bufsz) return NULL;
    memcpy(buf, p, len);
    buf[len] = '\0';
    return buf;
}

static void add_caller_entry(struct hot_func *hf, const char *name, double pct) {
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
    hf->callers = realloc(hf->callers,
        (size_t)(hf->n_callers + 1) * sizeof(*hf->callers));
    hf->callers[hf->n_callers].name = strdup(name);
    hf->callers[hf->n_callers].pct = pct;
    hf->n_callers++;
}

static int parse_callers(struct perf_opts *opts, struct perf_profile *prof) {
    if (prof->n_funcs == 0) return 0;

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "perf report --stdio -g caller --no-children "
        "--percent-limit 0.5 -i '%s/perf.data' 2>/dev/null",
        g_tmpdir);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    if (!out) return -1;

    /*
     * perf report -g caller output formats:
     *
     * Multi-caller (branching at depth-1):
     *   XX.XX%  cmd  obj  [.] FUNC
     *           |--10.22%--bench_full_response
     *           |          hpack_decode (inlined)
     *           |          ...
     *            --8.57%--bench_decode
     *                      hpack_decode (inlined)
     *
     * Single-chain (100% from one path):
     *   XX.XX%  cmd  obj  [.] FUNC
     *           ---_start
     *              __libc_start_main
     *              main
     *              bench_huffman_pair
     *              hpack_huff_decode (inlined)
     *
     * Strategy:
     * 1. Match function header [.] FUNC_NAME
     * 2. Look for depth-1 entries:
     *    a. |--XX%--NAME or --XX%--NAME → caller with percentage
     *    b. ---NAME → single chain (100%)
     * 3. For boring depth-1 names, follow chain to find better name.
     *    "Better" = last non-boring name before self or a sub-branch.
     * 4. Record column of first depth-1 to distinguish sub-branches.
     */
    int cur_func = -1;
    int depth1_col = -1;     /* column of '--' for depth-1 entries */
    int in_chain = 0;        /* inside a chain after depth-1 entry */
    double chain_pct = 0;    /* pct for current chain */
    char chain_best[256];    /* best (non-boring) name seen in chain */
    char nbuf[256];

    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        /* Function header: " XX.XX%  cmd  obj  [.] func_name" */
        char *bracket = strstr(line, "[.] ");
        if (bracket) {
            /* Flush any pending chain */
            if (in_chain && cur_func >= 0 && chain_best[0])
                add_caller_entry(&prof->funcs[cur_func],
                                 chain_best, chain_pct);
            cur_func = -1;
            depth1_col = -1;
            in_chain = 0;
            char *name = bracket + 4;
            size_t nlen = strlen(name);
            while (nlen > 0 &&
                   (name[nlen-1] == ' ' || name[nlen-1] == '\n'))
                nlen--;
            for (int f = 0; f < prof->n_funcs; f++) {
                if (strlen(prof->funcs[f].name) == nlen &&
                    strncmp(prof->funcs[f].name, name, nlen) == 0) {
                    cur_func = f;
                    break;
                }
            }
            goto next_line;
        }

        if (cur_func < 0) goto next_line;

        /* Blank line → end of this function's tree */
        {
            const char *t = line;
            while (*t == ' ' || *t == '\t') t++;
            if (*t == '\0' || *t == '\n') {
                if (in_chain && chain_best[0])
                    add_caller_entry(&prof->funcs[cur_func],
                                     chain_best, chain_pct);
                in_chain = 0;
                cur_func = -1;
                goto next_line;
            }
        }

        /* Check for percentage branch: |--XX.XX%--NAME or --XX.XX%--NAME */
        char *pctdash = strstr(line, "%--");
        if (pctdash) {
            /* Find the '--' before the percentage */
            char *dd = pctdash;
            while (dd > line && !(dd[-1] == '-' && dd[0] == '-')) dd--;
            if (dd > line) dd--;
            int col = (int)(dd - line);

            /* Parse percentage */
            double pct = 0;
            char *numstart = dd + 2; /* skip '--' */
            sscanf(numstart, "%lf", &pct);

            /* Only accept depth-1 entries (matching first column seen) */
            if (depth1_col < 0)
                depth1_col = col;

            if (col == depth1_col) {
                /* Flush previous chain */
                if (in_chain && chain_best[0])
                    add_caller_entry(&prof->funcs[cur_func],
                                     chain_best, chain_pct);
                /* Extract caller name after "%--" */
                char *cname = pctdash + 3;
                char *n = extract_tree_name(cname, nbuf, sizeof(nbuf));
                if (n) {
                    chain_pct = pct;
                    if (is_boring_caller(n)) {
                        chain_best[0] = '\0'; /* hope to find better below */
                        in_chain = 1;
                    } else {
                        snprintf(chain_best, sizeof(chain_best), "%s", n);
                        in_chain = 1;
                    }
                }
            }
            /* Sub-branches (col != depth1_col) are ignored */
            goto next_line;
        }

        /* Check for single-chain start: ---NAME (no percentage) */
        char *triple = strstr(line, "---");
        if (triple && !strstr(line, "%--")) {
            /* Flush previous chain */
            if (in_chain && chain_best[0])
                add_caller_entry(&prof->funcs[cur_func],
                                 chain_best, chain_pct);
            int col = (int)(triple - line);
            if (depth1_col < 0)
                depth1_col = col;
            char *n = extract_tree_name(triple + 3, nbuf, sizeof(nbuf));
            chain_pct = 100.0;
            chain_best[0] = '\0';
            if (n && !is_boring_caller(n))
                snprintf(chain_best, sizeof(chain_best), "%s", n);
            in_chain = 1;
            goto next_line;
        }

        /* Chain continuation line — just a function name.
           Update chain_best to the last non-boring, non-self name
           (closest to the hot function in the call chain). */
        if (in_chain) {
            char *n = extract_tree_name(line, nbuf, sizeof(nbuf));
            if (n && !is_boring_caller(n)) {
                /* Skip if this is the hot function itself
                   (match after stripping LTO suffixes) */
                char cn[256], cf[256];
                snprintf(cn, sizeof(cn), "%s", n);
                strip_compiler_suffix(cn);
                snprintf(cf, sizeof(cf), "%s",
                         prof->funcs[cur_func].name);
                strip_compiler_suffix(cf);
                if (strcmp(cn, cf) != 0)
                    snprintf(chain_best, sizeof(chain_best), "%s", n);
            }
        }

    next_line:
        line = nl ? nl + 1 : NULL;
    }

    /* Flush final chain */
    if (in_chain && cur_func >= 0 && chain_best[0])
        add_caller_entry(&prof->funcs[cur_func], chain_best, chain_pct);

    free(out);

    /* Sort callers by descending pct */
    for (int f = 0; f < prof->n_funcs; f++) {
        struct hot_func *hf = &prof->funcs[f];
        for (int i = 0; i < hf->n_callers - 1; i++)
            for (int j = i + 1; j < hf->n_callers; j++)
                if (hf->callers[j].pct > hf->callers[i].pct) {
                    struct caller_entry tmp = hf->callers[i];
                    hf->callers[i] = hf->callers[j];
                    hf->callers[j] = tmp;
                }
    }

    return 0;
}

/* ── Phase 2c: perf annotate ─────────────────────────────────────────── */

static int run_perf_annotate(struct perf_opts *opts,
                             struct perf_profile *prof) {
    int insn_cap = 64;
    prof->insns = malloc((size_t)insn_cap * sizeof(*prof->insns));
    prof->n_insns = 0;

    for (int f = 0; f < prof->n_funcs; f++) {
        const char *raw_name = prof->funcs[f].name;

        char cmd[4096];
        snprintf(cmd, sizeof(cmd),
            "perf annotate --stdio --symbol='%s' "
            "-i '%s/perf.data' 2>/dev/null",
            raw_name, g_tmpdir);

        int status;
        char *out = run_cmd(cmd, &status, opts->verbose);
        if (!out) continue;

        int func_insns = 0;
        uint32_t cur_source_line = 0;

        char *line = out;
        while (line && *line && func_insns < opts->insns_n) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            /* Try instruction line: " 12.34 :  401234:  insn..." */
            double pct;
            uint64_t addr;
            char asm_buf[512];
            int n = sscanf(line, " %lf : %" SCNx64 ": %511[^\n]",
                           &pct, &addr, asm_buf);
            if (n == 3 && pct >= 0.5) {
                if (prof->n_insns >= insn_cap) {
                    insn_cap *= 2;
                    prof->insns = realloc(prof->insns,
                        (size_t)insn_cap * sizeof(*prof->insns));
                }
                struct hot_insn *hi =
                    &prof->insns[prof->n_insns++];
                hi->func_name = strdup(prof->funcs[f].name);
                hi->addr = addr;
                hi->pct = pct;
                char *a = asm_buf;
                while (*a == ' ' || *a == '\t') a++;
                hi->asm_text = strdup(a);
                hi->source_line = cur_source_line;
                func_insns++;
            }

            /* Track source annotations: "  : /path/file.c:42" */
            char *colon;
            if ((colon = strstr(line, ".c:")) != NULL ||
                (colon = strstr(line, ".h:")) != NULL) {
                uint32_t sl;
                if (sscanf(colon + 3, "%u", &sl) == 1)
                    cur_source_line = sl;
            }

            line = nl ? nl + 1 : NULL;
        }
        free(out);
    }
    return 0;
}

/* ── Phase 2d: AMDuProf ──────────────────────────────────────────────── */

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

    /* Phase 1: collect with assess config (IPC + L1d + branches + misaligned) */
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "'%s' collect --config assess -o '%s' %s 2>&1",
        uprof_bin, uprof_dir, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);

    /* Extract data directory from collect output:
       "Generated data files path: /path/to/AMDuProf-xxx" */
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

    int cap = 16;
    prof->uprof_funcs = malloc((size_t)cap * sizeof(*prof->uprof_funcs));
    prof->n_uprof_funcs = 0;

    /* Find the "HOTTEST FUNCTIONS" section header line, then parse rows.
       CSV columns (assess config):
       FUNCTION, CYCLES, RETIRED_INST, IPC, CPI,
       BR_MISP(PTI), %BR_MISP, L1_DC_ACC(PTI), L1_DC_MISS(PTI),
       %L1_DC_MISS, L1_REFILL_LOCAL(PTI), L1_REFILL_REMOTE(PTI),
       MISALIGNED(PTI), Module */
    char line[4096];
    int in_funcs = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "HOTTEST FUNCTIONS")) {
            /* Next line is the column header — skip it */
            fgets(line, sizeof(line), fp);
            in_funcs = 1;
            continue;
        }
        if (!in_funcs) continue;

        /* Empty line ends the function table */
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            break;

        /* Extract quoted function name */
        if (line[0] != '"') break;
        char *name_end = strchr(line + 1, '"');
        if (!name_end) continue;
        *name_end = '\0';
        const char *func_name = line + 1;

        /* Match against known hot functions */
        int matched = 0;
        for (int f = 0; f < prof->n_funcs; f++) {
            if (strcmp(func_name, prof->funcs[f].name) == 0) {
                matched = 1;
                break;
            }
        }
        if (!matched) continue;

        /* Parse CSV fields after the function name:
           ,CYCLES,RETIRED_INST,IPC,CPI,BR_MISP_PTI,%BR_MISP,
           L1_DC_ACC_PTI,L1_DC_MISS_PTI,%L1_DC_MISS,
           L1_REFILL_LOCAL_PTI,L1_REFILL_REMOTE_PTI,MISALIGNED_PTI,Module */
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

        if (prof->n_uprof_funcs >= cap) {
            cap *= 2;
            prof->uprof_funcs = realloc(prof->uprof_funcs,
                (size_t)cap * sizeof(*prof->uprof_funcs));
        }
        struct uprof_func *uf =
            &prof->uprof_funcs[prof->n_uprof_funcs++];
        uf->name = strdup(func_name);
        uf->ipc = ipc;
        uf->l1_dc_miss_pct = l1_dc_miss_pct;
        uf->br_misp_pti = br_misp_pti;
        uf->misaligned_pti = misaligned_pti;
    }
    fclose(fp);
    return 0;
}

/* ── Phase 2e: Top-down analysis ─────────────────────────────────────── */

/* Intel: perf stat --topdown → retiring/bad_spec/fe_bound/be_bound
   AMD Zen: per-slot dispatch stall events → dispatched/backend/frontend */

static int run_topdown_intel(struct perf_opts *opts,
                             struct perf_profile *prof) {
    char cmd[4096];
    int status;
    snprintf(cmd, sizeof(cmd),
        "perf stat --topdown -x ',' -- %s 2>&1 1>/dev/null",
        opts->cmd_str);

    char *out = run_cmd(cmd, &status, opts->verbose);
    if (!out) return 0;

    struct topdown_metrics *td = &prof->topdown;
    memset(td, 0, sizeof(*td));
    td->level = 1;

    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        double val;
        char metric[128];
        if (sscanf(line, " %lf,,%127[^,]", &val, metric) >= 2 ||
            sscanf(line, "%lf,,%127[^,]", &val, metric) >= 2) {
            for (char *p = metric; *p; p++)
                *p = (*p >= 'A' && *p <= 'Z') ? *p + 32 : *p;

            if (strstr(metric, "retiring"))
                td->retiring = val;
            else if (strstr(metric, "bad") &&
                     strstr(metric, "spec"))
                td->bad_spec = val;
            else if (strstr(metric, "frontend") ||
                     strstr(metric, "fe_bound") ||
                     strstr(metric, "fe bound"))
                td->frontend = val;
            else if (strstr(metric, "backend") ||
                     strstr(metric, "be_bound") ||
                     strstr(metric, "be bound"))
                td->backend = val;
        }

        line = nl ? nl + 1 : NULL;
    }
    free(out);

    if (td->retiring > 0 || td->bad_spec > 0 ||
        td->frontend > 0 || td->backend > 0)
        prof->has_topdown = 1;

    return 0;
}

static int run_topdown_amd(struct perf_opts *opts,
                           struct perf_profile *prof) {
    char cmd[4096];
    int status;

    /* AMD Zen topdown: per-slot dispatch stall events.
       total_slots = dispatched + backend + frontend + smt_contention
       Percentages computed from slot fractions. */
    snprintf(cmd, sizeof(cmd),
        "perf stat -e "
        "de_src_op_disp.all:u,"
        "de_no_dispatch_per_slot.backend_stalls:u,"
        "de_no_dispatch_per_slot.no_ops_from_frontend:u,"
        "de_no_dispatch_per_slot.smt_contention:u "
        "-x ',' -- %s 2>&1 1>/dev/null",
        opts->cmd_str);

    char *out = run_cmd(cmd, &status, opts->verbose);
    if (!out) return 0;

    uint64_t dispatched = 0, backend = 0, frontend = 0, smt = 0;

    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        uint64_t val;
        char event[128];
        if (sscanf(line, "%" SCNu64 ",,%127[^,]", &val, event) >= 2) {
            if (strstr(event, "de_src_op_disp"))
                dispatched = val;
            else if (strstr(event, "backend"))
                backend = val;
            else if (strstr(event, "frontend") ||
                     strstr(event, "no_ops_from"))
                frontend = val;
            else if (strstr(event, "smt"))
                smt = val;
        }

        line = nl ? nl + 1 : NULL;
    }
    free(out);

    uint64_t total = dispatched + backend + frontend + smt;
    if (total == 0) return 0;

    struct topdown_metrics *td = &prof->topdown;
    memset(td, 0, sizeof(*td));
    td->level = 1;
    td->retiring = 100.0 * dispatched / total;
    td->backend  = 100.0 * backend / total;
    td->frontend = 100.0 * (frontend + smt) / total;
    /* bad_spec not cleanly separable on AMD (would need
       retired vs dispatched ops); report 0 */
    td->bad_spec = 0;
    prof->has_topdown = 1;

    return 0;
}

static int run_topdown(struct perf_opts *opts, struct perf_profile *prof) {
    if (opts->topdown_mode < 0) return 0;

    /* Try Intel --topdown first */
    int status;
    char *probe = run_cmd(
        "perf stat --topdown -- true 2>&1", &status, 0);
    int intel_ok = (probe && strstr(probe, "retiring") != NULL);
    free(probe);

    if (intel_ok)
        return run_topdown_intel(opts, prof);

    /* Try AMD Zen per-slot dispatch stall events */
    probe = run_cmd(
        "perf stat -e de_no_dispatch_per_slot.backend_stalls:u "
        "-- true 2>&1", &status, 0);
    int amd_ok = (status == 0 && probe &&
                  !strstr(probe, "not supported") &&
                  !strstr(probe, "<not counted>"));
    free(probe);

    if (amd_ok)
        return run_topdown_amd(opts, prof);

    if (opts->topdown_mode > 0) {
        fprintf(stderr,
            "error: --topdown specified but no topdown events "
            "available (need Intel Icelake+ or AMD Zen)\n");
        return -1;
    }
    return 0;
}

/* ── Phase 2f: Cache-miss attribution ────────────────────────────────── */

static int run_cache_misses(struct perf_opts *opts,
                            struct perf_profile *prof) {
    if (opts->cachemiss_mode < 0) return 0;

    /* perf is already confirmed available; just record cache-misses */
    char cm_data[PATH_MAX];
    snprintf(cm_data, sizeof(cm_data), "%s/cachemiss.data", g_tmpdir);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "perf record -e cache-misses -c 1000 "
        "-o '%s' -- %s 2>&1 1>/dev/null",
        cm_data, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    free(out);

    if (access(cm_data, F_OK) != 0) {
        if (opts->cachemiss_mode > 0)
            fprintf(stderr, "error: cache-miss recording failed\n");
        return opts->cachemiss_mode > 0 ? -1 : 0;
    }

    int cap = 64;
    prof->cm_sites = malloc((size_t)cap * sizeof(*prof->cm_sites));
    prof->n_cm_sites = 0;

    int limit = prof->n_funcs < opts->top_n ?
                prof->n_funcs : opts->top_n;
    for (int f = 0; f < limit; f++) {
        const char *raw_name = prof->funcs[f].name;

        snprintf(cmd, sizeof(cmd),
            "perf annotate --stdio --symbol='%s' "
            "-i '%s' 2>/dev/null",
            raw_name, cm_data);

        out = run_cmd(cmd, &status, opts->verbose);
        if (!out) continue;

        uint32_t cur_source_line = 0;
        char *line = out;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            double pct;
            uint64_t addr;
            char asm_buf[512];
            int n = sscanf(line, " %lf : %" SCNx64 ": %511[^\n]",
                           &pct, &addr, asm_buf);
            if (n == 3 && pct >= 0.05) {
                if (prof->n_cm_sites >= cap) {
                    cap *= 2;
                    prof->cm_sites = realloc(prof->cm_sites,
                        (size_t)cap * sizeof(*prof->cm_sites));
                }
                struct cache_miss_site *cm =
                    &prof->cm_sites[prof->n_cm_sites++];
                cm->func_name = strdup(prof->funcs[f].name);
                cm->source_line = cur_source_line;
                cm->pct = pct;
                char *a = asm_buf;
                while (*a == ' ' || *a == '\t') a++;
                cm->asm_text = strdup(a);
            }

            /* Track source line annotations */
            char *colon;
            if ((colon = strstr(line, ".c:")) != NULL ||
                (colon = strstr(line, ".h:")) != NULL) {
                uint32_t sl;
                if (sscanf(colon + 3, "%u", &sl) == 1)
                    cur_source_line = sl;
            }

            line = nl ? nl + 1 : NULL;
        }
        free(out);
    }

    return 0;
}

/* ── Phase 2g: llvm-mca throughput analysis ──────────────────────────── */

static int run_mca(struct perf_opts *opts, struct perf_profile *prof) {
    if (opts->mca_mode < 0) return 0;
    if (!has_tool("llvm-mca")) {
        if (opts->mca_mode > 0) {
            fprintf(stderr,
                "error: --mca specified but llvm-mca not found\n");
            return -1;
        }
        return 0;
    }

    int cap = 16;
    prof->mca_blocks = malloc((size_t)cap * sizeof(*prof->mca_blocks));
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
    for (int f = 0; f < limit; f++) {
        const char *raw_name = prof->funcs[f].name;

        /* Extract disassembly for this function — use raw name
           (may include .constprop.N/.isra.N from LTO) */
        char cmd[4096];
        snprintf(cmd, sizeof(cmd),
            "objdump -d --no-show-raw-insn '%s' 2>/dev/null | "
            "awk '/^[0-9a-f]+ <%s>:$/,/^$/' ",
            opts->binary_path, raw_name);

        char *disas = run_cmd(cmd, &status, opts->verbose);
        if (!disas || !*disas) { free(disas); continue; }

        /* Write cleaned instructions to temp file:
           strip addresses and labels, keep only mnemonics */
        char tmpasm[PATH_MAX];
        snprintf(tmpasm, sizeof(tmpasm), "%s/mca_block.s", g_tmpdir);
        FILE *fp = fopen(tmpasm, "w");
        if (!fp) { free(disas); continue; }

        int n_insns = 0;
        char *line = disas;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            /* Instruction lines look like: "  401234:  mov ..." or
               "  401234:\tmov ..." — have a colon followed by insn */
            char *colon = strchr(line, ':');
            if (colon && colon != line) {
                char *insn = colon + 1;
                while (*insn == ' ' || *insn == '\t') insn++;
                /* Skip empty lines, labels, directives */
                if (*insn && *insn != '<' && *insn != '.' &&
                    *insn != '#' && *insn != ';') {
                    /* Strip objdump annotations for llvm-mca:
                       <symbol+offset>, # addr <symbol>,
                       and bare hex jump targets */
                    char cleaned[512];
                    int ci = 0;
                    for (const char *s = insn;
                         *s && ci < (int)sizeof(cleaned) - 1; ) {
                        if (*s == '<') {
                            while (*s && *s != '>') s++;
                            if (*s == '>') s++;
                        } else if (*s == '#') {
                            break;
                        } else {
                            cleaned[ci++] = *s++;
                        }
                    }
                    /* trim trailing spaces */
                    while (ci > 0 && cleaned[ci-1] == ' ') ci--;
                    cleaned[ci] = '\0';
                    /* Replace bare hex jump targets with labels:
                       "jg     505a" → "jg .Ltmp" */
                    if (ci > 0 && (cleaned[0] == 'j' ||
                        (ci > 4 && strncmp(cleaned, "call", 4) == 0))) {
                        char *sp = cleaned;
                        while (*sp && *sp != ' ' && *sp != '\t') sp++;
                        while (*sp == ' ' || *sp == '\t') sp++;
                        /* Check if operand is a bare hex number */
                        if (*sp && *sp != '*' && *sp != '%' &&
                            *sp != '(' && *sp != '.') {
                            int all_hex = 1;
                            for (char *h = sp; *h; h++) {
                                if (!((*h >= '0' && *h <= '9') ||
                                      (*h >= 'a' && *h <= 'f') ||
                                      (*h >= 'A' && *h <= 'F'))) {
                                    all_hex = 0;
                                    break;
                                }
                            }
                            if (all_hex && sp > cleaned) {
                                strcpy(sp, ".Ltmp");
                                ci = (int)(sp - cleaned) + 5;
                            }
                        }
                    }
                    if (ci > 0) {
                        fprintf(fp, "%s\n", cleaned);
                        n_insns++;
                    }
                }
            }

            line = nl ? nl + 1 : NULL;
        }
        fclose(fp);
        free(disas);

        if (n_insns == 0) continue;

        /* Run llvm-mca */
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

        line = mca_out;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            if (strstr(line, "Block RThroughput"))
                sscanf(strstr(line, ":") + 1, " %lf", &rthroughput);
            else if (strstr(line, "Total uOps"))
                sscanf(strstr(line, ":") + 1, " %d", &uops);
            else if (strstr(line, "IPC") && !strstr(line, "Block"))
                sscanf(strstr(line, ":") + 1, " %lf", &ipc);
            else if (strstr(line, "Bottleneck")) {
                char *p = strstr(line, ":");
                if (p) {
                    p++;
                    while (*p == ' ') p++;
                    snprintf(bottleneck, sizeof(bottleneck), "%s", p);
                    /* Trim trailing whitespace */
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
            if (prof->n_mca_blocks >= cap) {
                cap *= 2;
                prof->mca_blocks = realloc(prof->mca_blocks,
                    (size_t)cap * sizeof(*prof->mca_blocks));
            }
            struct mca_block *mb =
                &prof->mca_blocks[prof->n_mca_blocks++];
            mb->func_name = strdup(prof->funcs[f].name);
            mb->block_rthroughput = rthroughput;
            mb->ipc = ipc;
            mb->n_uops = uops;
            mb->bottleneck = bottleneck[0] ?
                strdup(bottleneck) : NULL;
        }
    }

    return 0;
}

/* ── Phase 3b: pahole struct layout analysis ─────────────────────────── */

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
    prof->layouts = malloc((size_t)cap * sizeof(*prof->layouts));
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

        /* Find all "TYPE *" patterns (handles both struct X * and
           typedef'd names like hpack_ctx *) */
        const char *p = sig;
        while (*p) {
            /* Skip to an identifier character */
            while (*p && !(*p >= 'a' && *p <= 'z') &&
                   !(*p >= 'A' && *p <= 'Z') && *p != '_')
                p++;
            if (!*p) break;

            /* Extract identifier */
            char type_name[256];
            int ti = 0;
            const char *start = p;
            while (*p && ((*p >= 'a' && *p <= 'z') ||
                          (*p >= 'A' && *p <= 'Z') ||
                          (*p >= '0' && *p <= '9') || *p == '_') &&
                   ti < (int)sizeof(type_name) - 1)
                type_name[ti++] = *p++;
            type_name[ti] = '\0';

            /* Check if followed by " *" (pointer parameter) */
            const char *q = p;
            while (*q == ' ') q++;
            if (*q != '*') { (void)start; continue; }

            /* Skip known primitive types */
            int skip = 0;
            for (int s = 0; skip_types[s]; s++) {
                if (strcmp(type_name, skip_types[s]) == 0) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            /* For "struct X", extract just X */
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

            /* Check for duplicates */
            int dup = 0;
            for (int i = 0; i < prof->n_layouts; i++) {
                if (strcmp(prof->layouts[i].type_name,
                           probe_name) == 0) {
                    dup = 1;
                    break;
                }
            }
            if (dup) continue;

            /* Run pahole */
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

            /* Parse: "size: N, cachelines: N ..." */
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

            /* Only include interesting types */
            if (holes == 0 && cachelines <= 1) continue;

            if (prof->n_layouts >= cap) {
                cap *= 2;
                prof->layouts = realloc(prof->layouts,
                    (size_t)cap * sizeof(*prof->layouts));
            }
            struct struct_layout *sl =
                &prof->layouts[prof->n_layouts++];
            sl->type_name = strdup(probe_name);
            sl->size = size;
            sl->holes = holes;
            sl->padding = padding;
            sl->cachelines = cachelines;
            sl->func_name = strdup(prof->funcs[f].name);
        }
    }

    return 0;
}

/* ── Phase 3: Skeleton cross-reference ───────────────────────────────── */

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

    /* Match hot functions → collected symbols.
       When multiple files define the same symbol (e.g. main), prefer
       the file whose basename is a substring of the binary name. */
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

                /* Score: prefer source file related to binary name */
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
            hf->skeleton_sig = strdup(best_sym->text);
            hf->source_file = strdup(best_fe->abs_path);
            hf->start_line = best_sym->start_line;
            hf->end_line = best_sym->end_line;

            if (best_sym->n_callees > 0) {
                hf->n_callees = best_sym->n_callees;
                hf->callees = malloc(
                    (size_t)best_sym->n_callees * sizeof(char *));
                for (int k = 0; k < best_sym->n_callees; k++)
                    hf->callees[k] =
                        strdup(best_sym->callees[k]);
            }
        }
    }

    ts_query_delete(query);
    ts_parser_delete(parser);
}

/* ── Phase 3c: Compiler optimization remarks ─────────────────────────── */

static int run_remarks(struct perf_opts *opts, struct perf_profile *prof) {
    if (opts->remarks_mode < 0) return 0;

    /* Detect compiler */
    int status;
    char *cc_out = run_cmd("cc --version 2>&1", &status, 0);
    int is_gcc = 0, is_clang = 0;
    if (cc_out) {
        /* Lowercase scan */
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

    int cap = 64;
    prof->remarks = malloc((size_t)cap * sizeof(*prof->remarks));
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
                ccjson = malloc((size_t)flen + 1);
                fread(ccjson, 1, (size_t)flen, fp);
                ccjson[flen] = '\0';
            }
            fclose(fp);
        }
    }

    for (int s = 0; s < n_src; s++) {
        /* Build compile command for this source file */
        char cmd[8192];
        int found_cmd = 0;

        /* Try compile_commands.json first */
        if (ccjson) {
            const char *basename = strrchr(src_files[s], '/');
            basename = basename ? basename + 1 : src_files[s];

            /* Find "file":"...basename" entry */
            char needle[PATH_MAX];
            snprintf(needle, sizeof(needle), "\"file\":\"%s\"", basename);
            /* Also try with path components */
            char *entry = strstr(ccjson, needle);
            if (!entry) {
                snprintf(needle, sizeof(needle), "\"%s\"", basename);
                /* Search for file field containing our basename */
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
                            /* Check if this file entry matches */
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
                /* Find "command" in same object (search backwards/forwards) */
                char *cmd_field = NULL;
                /* Search forward within ~2KB */
                char *search = entry;
                for (int tries = 0; tries < 2000 && *search; tries++, search++) {
                    if (strncmp(search, "\"command\"", 9) == 0) {
                        cmd_field = search;
                        break;
                    }
                }
                /* Also search backward */
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
                            /* Modify: replace -o ... with -S -o /dev/null,
                               append remark flags */
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

        /* Fallback: standalone compilation */
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

        /* Parse remark lines: file:line:col: category: message */
        char *line = out;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';

            /* Match: "file:line:col: note/optimized/remark: ..." for both
               GCC and Clang formats */
            char rfile[PATH_MAX];
            uint32_t rline;
            uint32_t rcol;
            char category[64];
            char message[512];

            int matched = 0;
            /* GCC: file:line:col: optimized: message
               Clang: file:line:col: remark: message */
            if (sscanf(line, "%[^:]:%u:%u: %63[^:]: %511[^\n]",
                       rfile, &rline, &rcol, category, message) == 5)
                matched = 1;

            if (matched) {
                /* Normalize category */
                char *cat = category;
                while (*cat == ' ') cat++;
                /* Trim trailing spaces */
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

                /* Check if this line belongs to any hot function */
                for (int f = 0; f < prof->n_funcs; f++) {
                    struct hot_func *hf = &prof->funcs[f];
                    if (!hf->source_file) continue;

                    /* Match source file */
                    const char *hf_base = strrchr(hf->source_file, '/');
                    hf_base = hf_base ? hf_base + 1 : hf->source_file;
                    const char *rf_base = strrchr(rfile, '/');
                    rf_base = rf_base ? rf_base + 1 : rfile;
                    if (strcmp(hf_base, rf_base) != 0) continue;

                    /* Check line range */
                    if (rline < hf->start_line || rline > hf->end_line)
                        continue;

                    /* Count existing remarks for this function */
                    int func_remarks = 0;
                    for (int r = 0; r < prof->n_remarks; r++) {
                        if (strcmp(prof->remarks[r].func_name,
                                   hf->name) == 0)
                            func_remarks++;
                    }
                    if (func_remarks >= 10) break;

                    if (prof->n_remarks >= cap) {
                        cap *= 2;
                        prof->remarks = realloc(prof->remarks,
                            (size_t)cap * sizeof(*prof->remarks));
                    }
                    struct remark_entry *re =
                        &prof->remarks[prof->n_remarks++];
                    re->func_name = strdup(hf->name);
                    re->source_file = strdup(hf->source_file);
                    re->line = rline;
                    re->category = strdup(norm_cat);
                    re->message = strdup(message);
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

/* ── Phase 4: Output ─────────────────────────────────────────────────── */

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
    printf("cycles: %s  insn: %s  IPC: %.2f  wall: %.2fs\n",
           c_cyc, c_insn, s->ipc, s->wall_seconds);
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

    /* Hot instructions */
    if (prof->n_insns > 0) {
        printf("--- hot instructions ---\n");
        const char *cur_func = NULL;
        for (int i = 0; i < prof->n_insns; i++) {
            struct hot_insn *hi = &prof->insns[i];
            if (!cur_func ||
                strcmp(cur_func, hi->func_name) != 0) {
                cur_func = hi->func_name;
                const char *src = NULL;
                for (int f = 0; f < prof->n_funcs; f++) {
                    if (strcmp(prof->funcs[f].name,
                              cur_func) == 0 &&
                        prof->funcs[f].source_file) {
                        src = short_path(
                            prof->funcs[f].source_file,
                            opts->source_dir);
                        break;
                    }
                }
                if (src && hi->source_line)
                    printf("[%s]  //%s:%u\n",
                           cur_func, src, hi->source_line);
                else
                    printf("[%s]\n", cur_func);
            }
            printf("  %5.1f%%  %s\n", hi->pct, hi->asm_text);
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

    /* Cache misses */
    if (prof->n_cm_sites > 0) {
        printf("--- cache misses ---\n");
        const char *cur_func = NULL;
        for (int i = 0; i < prof->n_cm_sites; i++) {
            struct cache_miss_site *cm = &prof->cm_sites[i];
            if (!cur_func ||
                strcmp(cur_func, cm->func_name) != 0) {
                cur_func = cm->func_name;
                const char *src = NULL;
                for (int f = 0; f < prof->n_funcs; f++) {
                    if (strcmp(prof->funcs[f].name,
                              cur_func) == 0 &&
                        prof->funcs[f].source_file) {
                        src = short_path(
                            prof->funcs[f].source_file,
                            opts->source_dir);
                        break;
                    }
                }
                if (src && cm->source_line)
                    printf("[%s]  //%s:%u\n",
                           cur_func, src, cm->source_line);
                else
                    printf("[%s]\n", cur_func);
            }
            printf("  %5.1f%%  %s\n", cm->pct, cm->asm_text);
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
                /* Find source info */
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

static int run_pipeline(struct perf_opts *opts, struct perf_profile *prof,
                        int has_debug) {
    fprintf(stderr, "profiling: %s\n", opts->cmd_str);

    if (run_perf_stat(opts, &prof->stats) != 0)
        fprintf(stderr, "warning: perf stat failed, continuing\n");

    run_topdown(opts, prof);

    if (run_perf_record(opts) != 0) {
        fprintf(stderr, "error: perf record failed\n");
        return -1;
    }

    parse_perf_report(opts, prof);
    parse_callers(opts, prof);
    run_perf_annotate(opts, prof);
    run_cache_misses(opts, prof);
    run_mca(opts, prof);
    run_uprof(opts, prof);
    xref_skeleton(opts, prof);
    run_remarks(opts, prof);
    run_pahole(opts, prof, has_debug);

    return 0;
}

/* ── A/B Comparison output ───────────────────────────────────────────── */

static void print_comparison(struct perf_opts *opts,
                              struct perf_profile *prof_a, const char *a_cmd,
                              struct perf_profile *prof_b, const char *b_cmd) {
    struct perf_stats *sa = &prof_a->stats;
    struct perf_stats *sb = &prof_b->stats;

    printf("=== perf A/B: %s vs %s ===\n", a_cmd, b_cmd);

    /* Stats CSV with deltas */
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
        printf("cycles,%s,%s,%s%s,%.1f%%\n", ca, cb,
               sb->cycles < sa->cycles ? "-" : "+", cd, pct);
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
        printf("insns,%s,%s,%s%s,%.1f%%\n", ia, ib,
               sb->instructions < sa->instructions ? "-" : "+", id, pct);
    }
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
    printf("\n");

    /* Hot function delta — merge by name, with caller info */
    printf("--- hot functions (A -> B) ---\n");
    printf("  A%%    B%%   delta  function\n");

    /* Print functions in B, matching against A */
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

    /* Per-function cache miss hotspots — top 3 per matched function */
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
            int count = 0;
            for (int i = 0; i < prof_a->n_cm_sites && count < 3; i++) {
                char cmn[256];
                snprintf(cmn, sizeof(cmn), "%s",
                         prof_a->cm_sites[i].func_name);
                strip_compiler_suffix(cmn);
                if (strcmp(cmn, bclean) == 0) {
                    printf("  A: %5.1f%%  %s\n",
                           prof_a->cm_sites[i].pct,
                           prof_a->cm_sites[i].asm_text);
                    count++;
                }
            }
            count = 0;
            for (int i = 0; i < prof_b->n_cm_sites && count < 3; i++) {
                char cmn[256];
                snprintf(cmn, sizeof(cmn), "%s",
                         prof_b->cm_sites[i].func_name);
                strip_compiler_suffix(cmn);
                if (strcmp(cmn, bclean) == 0) {
                    printf("  B: %5.1f%%  %s\n",
                           prof_b->cm_sites[i].pct,
                           prof_b->cm_sites[i].asm_text);
                    count++;
                }
            }
        }
        printf("\n");
    }

    /* Hot instruction summary — top 3 per matched function */
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
            int count = 0;
            for (int i = 0; i < prof_a->n_insns && count < 3; i++) {
                char iclean[256];
                snprintf(iclean, sizeof(iclean), "%s",
                         prof_a->insns[i].func_name);
                strip_compiler_suffix(iclean);
                if (strcmp(iclean, bclean) == 0) {
                    printf("  A: %5.1f%%  %s\n",
                           prof_a->insns[i].pct,
                           prof_a->insns[i].asm_text);
                    count++;
                }
            }
            count = 0;
            for (int i = 0; i < prof_b->n_insns && count < 3; i++) {
                char iclean[256];
                snprintf(iclean, sizeof(iclean), "%s",
                         prof_b->insns[i].func_name);
                strip_compiler_suffix(iclean);
                if (strcmp(iclean, bclean) == 0) {
                    printf("  B: %5.1f%%  %s\n",
                           prof_b->insns[i].pct,
                           prof_b->insns[i].asm_text);
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
        int *b_matched = calloc((size_t)prof_b->n_remarks, sizeof(int));
        int any_diff = 0;
        const char *cur_func = NULL;

        /* A-only remarks (lost or changed in B) */
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
        /* B-only remarks (new in B) */
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
    for (int i = 0; i < prof->n_funcs; i++) {
        free(prof->funcs[i].name);
        free(prof->funcs[i].skeleton_sig);
        free(prof->funcs[i].source_file);
        for (int k = 0; k < prof->funcs[i].n_callees; k++)
            free(prof->funcs[i].callees[k]);
        free(prof->funcs[i].callees);
        for (int k = 0; k < prof->funcs[i].n_callers; k++)
            free(prof->funcs[i].callers[k].name);
        free(prof->funcs[i].callers);
    }
    free(prof->funcs);

    for (int i = 0; i < prof->n_insns; i++) {
        free(prof->insns[i].func_name);
        free(prof->insns[i].asm_text);
    }
    free(prof->insns);

    for (int i = 0; i < prof->n_uprof_funcs; i++)
        free(prof->uprof_funcs[i].name);
    free(prof->uprof_funcs);

    for (int i = 0; i < prof->n_mca_blocks; i++) {
        free(prof->mca_blocks[i].func_name);
        free(prof->mca_blocks[i].bottleneck);
    }
    free(prof->mca_blocks);

    for (int i = 0; i < prof->n_cm_sites; i++) {
        free(prof->cm_sites[i].func_name);
        free(prof->cm_sites[i].asm_text);
    }
    free(prof->cm_sites);

    for (int i = 0; i < prof->n_layouts; i++) {
        free(prof->layouts[i].type_name);
        free(prof->layouts[i].func_name);
    }
    free(prof->layouts);

    for (int i = 0; i < prof->n_remarks; i++) {
        free(prof->remarks[i].func_name);
        free(prof->remarks[i].source_file);
        free(prof->remarks[i].category);
        free(prof->remarks[i].message);
    }
    free(prof->remarks);
}

/* ── Entry point ─────────────────────────────────────────────────────── */

int perf_main(int argc, char *argv[]) {
    struct perf_opts opts = {
        .top_n      = 10,
        .insns_n    = 10,
        .runs       = 3,
        .build_cmd  = "make",
        .no_build   = 0,
        .uprof_mode = 0,
        .source_dir = ".",
        .verbose    = 0,
    };

    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "n:i:r:b:s:vh",
                              perf_long_options, NULL)) != -1) {
        switch (opt) {
        case 'n': opts.top_n    = atoi(optarg); break;
        case 'i': opts.insns_n  = atoi(optarg); break;
        case 'r': opts.runs     = atoi(optarg); break;
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

    /* Temp directory for perf data */
    snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/archmap-XXXXXX");
    if (!mkdtemp(g_tmpdir)) {
        perror("mkdtemp");
        free(opts.cmd_str);
        free(opts.binary_path);
        return 1;
    }
    atexit(cleanup_tmpdir);

    /* Phase 1: Build */
    if (phase_build(&opts) != 0) goto fail;

    /* Check prerequisites */
    if (check_perf_access() != 0) goto fail;

    if (!has_tool("perf")) {
        fprintf(stderr, "error: 'perf' not found in PATH\n");
        goto fail;
    }

    if (opts.vs_binary) {
        /* ── A/B comparison mode ─────────────────────────────── */
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

        /* Build A command: a_binary + B's trailing args */
        char **a_argv = malloc((size_t)opts.cmd_argc * sizeof(char *));
        a_argv[0] = a_binary;
        for (int i = 1; i < opts.cmd_argc; i++)
            a_argv[i] = opts.cmd_argv[i];
        char *a_cmd_str = join_argv(a_argv, opts.cmd_argc);

        /* Save B's state */
        char *b_cmd_str = opts.cmd_str;
        char *b_binary = opts.binary_path;

        /* Run A pipeline */
        char a_dir[PATH_MAX];
        snprintf(a_dir, sizeof(a_dir), "%s/a", top_tmpdir);
        mkdir(a_dir, 0700);
        strcpy(g_tmpdir, a_dir);

        opts.cmd_str = a_cmd_str;
        opts.binary_path = a_binary;
        int saved_no_build = opts.no_build;
        opts.no_build = 1;  /* A is pre-built */

        int has_debug_a = check_debug_symbols(a_binary);
        struct perf_profile prof_a;
        memset(&prof_a, 0, sizeof(prof_a));

        fprintf(stderr, "=== A (baseline): %s ===\n", a_cmd_str);
        int a_ok = run_pipeline(&opts, &prof_a, has_debug_a);

        /* Run B pipeline */
        char b_dir[PATH_MAX];
        snprintf(b_dir, sizeof(b_dir), "%s/b", top_tmpdir);
        mkdir(b_dir, 0700);
        strcpy(g_tmpdir, b_dir);

        opts.cmd_str = b_cmd_str;
        opts.binary_path = b_binary;
        opts.no_build = saved_no_build;

        int has_debug_b = check_debug_symbols(b_binary);
        struct perf_profile prof_b;
        memset(&prof_b, 0, sizeof(prof_b));

        fprintf(stderr, "=== B (new): %s ===\n", b_cmd_str);
        int b_ok = run_pipeline(&opts, &prof_b, has_debug_b);

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
    int has_debug = check_debug_symbols(opts.binary_path);
    if (!has_debug)
        fprintf(stderr,
            "warning: no debug symbols in %s "
            "(build with -g for source annotations)\n",
            opts.binary_path);

    struct perf_profile prof;
    memset(&prof, 0, sizeof(prof));

    if (run_pipeline(&opts, &prof, has_debug) != 0)
        goto fail;

    print_report(&opts, &prof);

    free_profile(&prof);
    free(opts.cmd_str);
    free(opts.binary_path);
    return 0;

fail:
    free(opts.cmd_str);
    free(opts.binary_path);
    return 1;
}
