#ifndef PERF_ANALYSIS_H
#define PERF_ANALYSIS_H

#include <stdint.h>
#include <fts.h>
#include <tree_sitter/api.h>

extern const TSLanguage *tree_sitter_c(void);

/* ── Shared structs (used by both main.c and perf_analysis.c) ──────── */

struct symbol {
    char *text;
    char **callees;
    int n_callees;
    int cap_callees;
    int is_fwd_decl;
    char *tag_name;
    uint32_t cap_idx;
    uint32_t start_line;
    uint32_t end_line;
    int section;
};

struct file_entry {
    char *abs_path;
    char *rel_path;
    struct symbol *syms;
    int n_syms;
    int cap_syms;
    char **includes;
    int n_includes;
};

/* ── Globals shared from main.c ────────────────────────────────────── */

extern struct file_entry *g_files;
extern int g_n_files, g_cap_files;
extern const char *QUERY_SOURCE;
extern int opt_show_calls;
extern char **g_file_tags;
extern int g_n_file_tags, g_cap_file_tags;

/* ── Functions shared from main.c ──────────────────────────────────── */

char *read_file_source(const char *path, long *out_length);
int should_skip_fts_entry(FTSENT *ent);
int is_source_file(const char *filename);
struct file_entry *add_file(const char *abs_path);
struct symbol *add_symbol(struct file_entry *fe);
char *skeleton_text(const char *source, TSNode node, uint32_t cap_idx);
void collect_callees(TSNode node, const char *source, struct symbol *sym);
void collect_file(const char *path, TSParser *parser, TSQuery *query,
                  char **defines, int n_defines,
                  char **search_paths, int n_search);

/* ── Perf data structures ──────────────────────────────────────────── */

struct perf_stats {
    uint64_t cycles, instructions;
    uint64_t cache_refs, cache_misses;
    uint64_t branches, branch_misses;
    double wall_seconds;
    double ipc, cache_miss_pct, branch_miss_pct;
};

struct hot_func {
    char *name;
    double overhead_pct;
    uint64_t samples;
    char *skeleton_sig;
    char *source_file;
    uint32_t start_line, end_line;
    char **callees;
    int n_callees;
};

struct hot_insn {
    char *func_name;
    uint64_t addr;
    double pct;
    char *asm_text;
    uint32_t source_line;
};

struct uprof_func {
    char *name;
    double cpi;
    uint64_t dc_misses, ic_misses;
};

struct topdown_metrics {
    double retiring, bad_spec, frontend, backend;
    int level;                /* 1 or 2 depending on kernel support */
};

struct mca_block {
    char *func_name;
    double block_rthroughput; /* cycles per iteration */
    double ipc;
    int n_uops;
    char *bottleneck;         /* e.g. "RetireOOO", "Dispatch" */
};

struct cache_miss_site {
    char *func_name;
    uint32_t source_line;
    double pct;               /* % of total cache-miss samples */
    char *asm_text;
};

struct struct_layout {
    char *type_name;
    uint32_t size;
    uint32_t holes;
    uint32_t padding;
    uint32_t cachelines;
    char *func_name;          /* hot function that uses this type */
};

struct perf_profile {
    struct perf_stats stats;
    struct hot_func *funcs;
    int n_funcs;
    struct hot_insn *insns;
    int n_insns;
    struct uprof_func *uprof_funcs;
    int n_uprof_funcs;
    struct topdown_metrics topdown;
    int has_topdown;
    struct mca_block *mca_blocks;    int n_mca_blocks;
    struct cache_miss_site *cm_sites; int n_cm_sites;
    struct struct_layout *layouts;    int n_layouts;
};

/* ── Entry point ───────────────────────────────────────────────────── */

int perf_main(int argc, char *argv[]);

#endif /* PERF_ANALYSIS_H */
