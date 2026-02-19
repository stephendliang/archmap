#ifndef PERF_ANALYSIS_H
#define PERF_ANALYSIS_H

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <fts.h>
#include <tree_sitter/api.h>

#include "arena.h"

extern const TSLanguage *tree_sitter_c(void);

/* ── Shared structs (used by both main.c and perf_analysis.c) ──────── */

struct symbol {
    char *text;
    char **callees;
    char *tag_name;
    int n_callees;
    int cap_callees;
    uint32_t start_line;
    uint32_t end_line;
    unsigned is_fwd_decl : 1;
    unsigned section     : 3;  /* SEC_STRUCT..SEC_FUNCTIONS (0..4) */
    unsigned cap_idx     : 3;  /* CAP_FUNC_DEF..CAP_INCLUDE (0..7) */
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

struct caller_entry {
    char *name;
    double pct;
};

struct hot_func {
    char *name;
    double overhead_pct;
    uint64_t samples;
    char *skeleton_sig;
    char *source_file;
    char **callees;
    struct caller_entry *callers;
    uint32_t start_line, end_line;
    int n_callees;
    int n_callers;
};

struct hot_insn {
    char *func_name;
    uint64_t addr;
    double pct;
    char *asm_text;
    char *source_file;
    uint32_t source_line;
};

struct uprof_func {
    char *name;
    double ipc;
    double l1_dc_miss_pct;
    double br_misp_pti;      /* branch mispredicts per 1K insns */
    double misaligned_pti;   /* misaligned loads per 1K insns */
};

struct topdown_metrics {
    double retiring, bad_spec, frontend, backend;
    int level;                /* 1 or 2 depending on kernel support */
};

struct mca_block {
    char *func_name;
    double block_rthroughput; /* cycles per iteration */
    double ipc;
    char *bottleneck;         /* e.g. "RetireOOO", "Dispatch" */
    int n_uops;
};

struct cache_miss_site {
    char *func_name;
    double pct;               /* % of total cache-miss samples */
    char *asm_text;
    char *source_file;
    uint64_t data_addr;       /* data virtual address (0 if unavailable) */
    uint32_t source_line;
};

struct struct_layout {
    char *type_name;
    uint32_t size;
    uint32_t holes;
    uint32_t padding;
    uint32_t cachelines;
    char *func_name;          /* hot function that uses this type */
};

struct remark_entry {
    char *func_name;
    char *source_file;
    char *category;    /* "optimized" or "missed" */
    char *message;
    uint32_t line;
};

struct mem_hotspot {
    char *func_name;
    char *source_file;
    char *asm_text;
    uint64_t cache_line;    /* data_addr >> 6 */
    double pct;             /* % of total cache-miss samples with ADDR */
    uint32_t source_line;
    int n_samples;          /* raw sample count */
};

struct run_stats {
    double mean, stddev;
    double *values;         /* raw values from each run */
    int n;
};

/* ── Profile data (moved from perf_analysis.c) ─────────────────────── */

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

/* ── Options (moved from perf_analysis.c) ──────────────────────────── */

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

/* ── Utility functions from perf_analysis.c ────────────────────────── */

extern char g_tmpdir[PATH_MAX];

char *run_cmd(const char *cmd, int *out_status, int verbose);
int has_tool(const char *name);
const char *find_tool(const char *name);
void fmt_count(char *buf, size_t sz, uint64_t val);
void strip_compiler_suffix(char *name);
int is_boring_caller(const char *name);
void add_caller_entry(struct perf_profile *prof, struct hot_func *hf,
                      const char *name, double pct);

/* ── Entry point ───────────────────────────────────────────────────── */

int perf_main(int argc, char *argv[]);

#endif /* PERF_ANALYSIS_H */
