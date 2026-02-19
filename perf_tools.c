/* perf_tools.c — External tool runners (AMDuProf, llvm-mca, pahole, remarks) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <fts.h>
#include <tree_sitter/api.h>
#include <capstone/capstone.h>

#include "arena.h"
#include "symres.h"
#include "perf_tools.h"
#include "perf_analysis.h"
#include "git_cache.h"

/* From perf_analysis.c */
extern char g_tmpdir[PATH_MAX];
char *run_cmd(const char *cmd, int *out_status, int verbose);
int has_tool(const char *name);
const char *find_tool(const char *name);
void strip_compiler_suffix(char *name);

/* ── AMDuProf (unchanged — vendor subprocess) ────────────────────────── */

int run_uprof(struct perf_opts *opts, struct perf_profile *prof) {
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

int run_mca(struct sym_resolver *sr, struct perf_opts *opts,
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

int run_pahole(struct perf_opts *opts, struct perf_profile *prof,
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

void xref_skeleton(struct perf_opts *opts,
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

int run_remarks(struct perf_opts *opts, struct perf_profile *prof) {
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
