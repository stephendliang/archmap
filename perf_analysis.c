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
    const char *source_dir;
    int verbose;
    char **cmd_argv;
    int cmd_argc;
    char *binary_path;
    char *cmd_str;
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
    if (!has_tool("AMDuProfCLI")) {
        if (opts->uprof_mode > 0) {
            fprintf(stderr,
                "error: --uprof specified but AMDuProfCLI not found\n");
            return -1;
        }
        return 0;
    }

    char uprof_dir[PATH_MAX];
    snprintf(uprof_dir, sizeof(uprof_dir), "%s/uprof", g_tmpdir);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "AMDuProfCLI collect --config cpi -o '%s' -- %s 2>&1",
        uprof_dir, opts->cmd_str);

    int status;
    char *out = run_cmd(cmd, &status, opts->verbose);
    free(out);
    if (status != 0) {
        if (opts->uprof_mode > 0)
            fprintf(stderr, "AMDuProfCLI collect failed\n");
        return opts->uprof_mode > 0 ? -1 : 0;
    }

    snprintf(cmd, sizeof(cmd),
        "AMDuProfCLI report -i '%s' --category cpu 2>&1", uprof_dir);
    out = run_cmd(cmd, &status, opts->verbose);
    if (!out) return 0;

    int cap = 16;
    prof->uprof_funcs = malloc((size_t)cap * sizeof(*prof->uprof_funcs));
    prof->n_uprof_funcs = 0;

    /* Parse report — look for known hot function names */
    int header_seen = 0;
    char *line = out;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        if (!header_seen) {
            if (strstr(line, "Function") && strstr(line, "CPI"))
                header_seen = 1;
            line = nl ? nl + 1 : NULL;
            continue;
        }

        for (int f = 0; f < prof->n_funcs; f++) {
            if (!strstr(line, prof->funcs[f].name)) continue;

            double cpi = 0;
            uint64_t dc = 0, ic = 0;
            char *p = strstr(line, prof->funcs[f].name);
            if (p) p += strlen(prof->funcs[f].name);

            /* Skip past commas to CPI field */
            int commas = 0;
            while (p && *p && commas < 2) {
                if (*p == ',') commas++;
                p++;
            }
            if (p)
                sscanf(p, " %lf , %" SCNu64 " , %" SCNu64,
                       &cpi, &dc, &ic);

            if (prof->n_uprof_funcs >= cap) {
                cap *= 2;
                prof->uprof_funcs = realloc(prof->uprof_funcs,
                    (size_t)cap * sizeof(*prof->uprof_funcs));
            }
            struct uprof_func *uf =
                &prof->uprof_funcs[prof->n_uprof_funcs++];
            uf->name = strdup(prof->funcs[f].name);
            uf->cpi = cpi;
            uf->dc_misses = dc;
            uf->ic_misses = ic;
            break;
        }

        line = nl ? nl + 1 : NULL;
    }
    free(out);
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

    /* Match hot functions → collected symbols */
    for (int f = 0; f < prof->n_funcs; f++) {
        struct hot_func *hf = &prof->funcs[f];
        char clean[256];
        snprintf(clean, sizeof(clean), "%s", hf->name);
        strip_compiler_suffix(clean);
        size_t clen = strlen(clean);

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

                hf->skeleton_sig = strdup(sym->text);
                hf->source_file = strdup(fe->abs_path);
                hf->start_line = sym->start_line;
                hf->end_line = sym->end_line;

                if (sym->n_callees > 0) {
                    hf->n_callees = sym->n_callees;
                    hf->callees = malloc(
                        (size_t)sym->n_callees * sizeof(char *));
                    for (int k = 0; k < sym->n_callees; k++)
                        hf->callees[k] =
                            strdup(sym->callees[k]);
                }
                goto next_func;
            }
        }
        next_func:;
    }

    ts_query_delete(query);
    ts_parser_delete(parser);
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
        printf("--- uprof ---\n");
        for (int i = 0; i < prof->n_uprof_funcs; i++) {
            struct uprof_func *uf = &prof->uprof_funcs[i];
            printf("%s: CPI %.2f, L1d miss %" PRIu64
                   ", L1i miss %" PRIu64 "\n",
                   uf->name, uf->cpi,
                   uf->dc_misses, uf->ic_misses);
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
        for (int i = 0; i < prof->n_mca_blocks; i++) {
            struct mca_block *mb = &prof->mca_blocks[i];
            printf("%s: RThroughput %.2f cyc, %d uops, IPC %.2f",
                   mb->func_name, mb->block_rthroughput,
                   mb->n_uops, mb->ipc);
            if (mb->bottleneck)
                printf(" (%s)", mb->bottleneck);
            printf("\n");
        }
        printf("\n");
    }

    /* pahole data layout */
    if (prof->n_layouts > 0) {
        printf("--- data layout ---\n");
        for (int i = 0; i < prof->n_layouts; i++) {
            struct struct_layout *sl = &prof->layouts[i];
            printf("%s: %uB (%u cacheline%s)",
                   sl->type_name, sl->size,
                   sl->cachelines,
                   sl->cachelines != 1 ? "s" : "");
            if (sl->holes > 0)
                printf(", %u hole%s (%uB wasted)",
                       sl->holes,
                       sl->holes != 1 ? "s" : "",
                       sl->padding > 0 ? sl->padding :
                       sl->holes * 4);
            printf("  [used by %s]\n", sl->func_name);
        }
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

    int has_debug = check_debug_symbols(opts.binary_path);
    if (!has_debug)
        fprintf(stderr,
            "warning: no debug symbols in %s "
            "(build with -g for source annotations)\n",
            opts.binary_path);

    /* Phase 2: Profile */
    struct perf_profile prof;
    memset(&prof, 0, sizeof(prof));

    fprintf(stderr, "profiling: %s\n", opts.cmd_str);

    if (run_perf_stat(&opts, &prof.stats) != 0)
        fprintf(stderr,
            "warning: perf stat failed, continuing\n");

    run_topdown(&opts, &prof);

    if (run_perf_record(&opts) != 0) {
        fprintf(stderr, "error: perf record failed\n");
        goto fail;
    }

    parse_perf_report(&opts, &prof);
    run_perf_annotate(&opts, &prof);

    run_cache_misses(&opts, &prof);

    run_mca(&opts, &prof);

    run_uprof(&opts, &prof);

    /* Phase 3: Skeleton cross-reference */
    xref_skeleton(&opts, &prof);

    run_pahole(&opts, &prof, has_debug);

    /* Phase 4: Output */
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
