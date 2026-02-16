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

/* ── Options ─────────────────────────────────────────────────────────── */

struct perf_opts {
    int top_n;
    int insns_n;
    int runs;
    const char *build_cmd;
    int no_build;
    int uprof_mode;       /* 1=force, -1=skip, 0=auto */
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
};

static struct option perf_long_options[] = {
    {"top",       required_argument, NULL, 'n'},
    {"insns",     required_argument, NULL, 'i'},
    {"runs",      required_argument, NULL, 'r'},
    {"build-cmd", required_argument, NULL, 'b'},
    {"no-build",  no_argument,       NULL, PERF_OPT_NO_BUILD},
    {"uprof",     no_argument,       NULL, PERF_OPT_UPROF},
    {"no-uprof",  no_argument,       NULL, PERF_OPT_NO_UPROF},
    {"source",    required_argument, NULL, 's'},
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

static char *join_argv(char **argv, int argc) {
    size_t total = 0;
    for (int i = 0; i < argc; i++)
        total += strlen(argv[i]) + 4;
    char *cmd = malloc(total + 1);
    char *p = cmd;
    for (int i = 0; i < argc; i++) {
        if (i > 0) *p++ = ' ';
        size_t al = strlen(argv[i]);
        memcpy(p, argv[i], al);
        p += al;
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
    if (status != 0) {
        fprintf(stderr, "perf record failed (exit %d)\n", status);
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
        char clean_name[256];
        snprintf(clean_name, sizeof(clean_name), "%s",
                 prof->funcs[f].name);
        strip_compiler_suffix(clean_name);

        char cmd[4096];
        snprintf(cmd, sizeof(cmd),
            "perf annotate --stdio --symbol='%s' "
            "-i '%s/perf.data' 2>/dev/null",
            clean_name, g_tmpdir);

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

    /* Walk source directory */
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
                collect_file(ent->fts_path, parser, query,
                             NULL, 0, NULL, 0);
            }
        }
        fts_close(ftsp);
    }
    free(src_dup);

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
    printf("cache: %.1f%% miss (%s/%s)  branch: %.1f%% miss (%s/%s)\n\n",
           s->cache_miss_pct, c_cm, c_cr,
           s->branch_miss_pct, c_bm, c_br);

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
        case PERF_OPT_UPROF:    opts.uprof_mode = 1;  break;
        case PERF_OPT_NO_UPROF: opts.uprof_mode = -1; break;
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

    if (run_perf_record(&opts) != 0) {
        fprintf(stderr, "error: perf record failed\n");
        goto fail;
    }

    parse_perf_report(&opts, &prof);
    run_perf_annotate(&opts, &prof);
    run_uprof(&opts, &prof);

    /* Phase 3: Skeleton cross-reference */
    xref_skeleton(&opts, &prof);

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
