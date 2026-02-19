/* perf_report.c — Statistics + report output for perf analysis */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <math.h>

#include "arena.h"
#include "perf_analysis.h"
#include "perf_report.h"

/* ── Source interleaving helpers ──────────────────────────────────────── */

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

static void print_source_line(const char *line) {
    while (*line == ' ' || *line == '\t') line++;
    while (*line && *line != '\n' && *line != '\r') {
        putchar(*line);
        line++;
    }
}

/* ── Statistics ───────────────────────────────────────────────────────── */

void compute_stats(struct run_stats *rs) {
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

static double betacf(double a, double b, double x) {
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 200; m++) {
        int m2 = 2 * m;
        double aa = (double)m * (b - (double)m) * x /
                    ((qam + (double)m2) * (a + (double)m2));
        d = 1.0 + aa * d; if (fabs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c; if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        h *= d * c;
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

static double t_cdf(double t_val, double df) {
    double x = df / (df + t_val * t_val);
    double ix = ibeta(x, df / 2.0, 0.5);
    if (t_val >= 0)
        return 1.0 - 0.5 * ix;
    else
        return 0.5 * ix;
}

double welch_t_test(struct run_stats *a, struct run_stats *b) {
    if (a->n < 2 || b->n < 2) return 1.0;
    double va = a->stddev * a->stddev, vb = b->stddev * b->stddev;
    double se = sqrt(va / a->n + vb / b->n);
    if (se < 1e-15) return (fabs(a->mean - b->mean) < 1e-15) ? 1.0 : 0.0;
    double t_val = (a->mean - b->mean) / se;
    double num = (va / a->n + vb / b->n);
    num *= num;
    double denom = (va * va) / ((double)a->n * a->n * (a->n - 1)) +
                   (vb * vb) / ((double)b->n * b->n * (b->n - 1));
    double df = denom > 0 ? num / denom : 2.0;
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

/* ── Report output ────────────────────────────────────────────────────── */

void print_report(struct perf_opts *opts,
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

/* ── A/B Comparison output ───────────────────────────────────────────── */

void print_comparison(struct perf_opts *opts,
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
