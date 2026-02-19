/* archmap.c - Generates a semantic skeleton of a C/C++ codebase */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fts.h>
#include <getopt.h>
#include <fnmatch.h>
#include <libgen.h>
#include <limits.h>
#include <tree_sitter/api.h>

extern const TSLanguage *tree_sitter_c(void);

#include "hmap/hmap_avx512.h"
#include "perf_analysis.h"
#include "git_cache.h"
#include "skeleton.h"
#include "output.h"

static int has_function_declarator(TSNode node) {
    const char *type = ts_node_type(node);
    if (strcmp(type, "function_declarator") == 0) return 1;
    if (strcmp(type, "pointer_declarator") == 0 ||
        strcmp(type, "parenthesized_declarator") == 0) {
        TSNode inner = ts_node_child_by_field_name(node, "declarator", 10);
        if (!ts_node_is_null(inner)) return has_function_declarator(inner);
    }
    return 0;
}

static int section_for_capture(uint32_t cap_idx, TSNode node) {
    switch (cap_idx) {
    case CAP_STRUCT_DEF: return SEC_STRUCT;
    case CAP_TYPEDEF:    return SEC_TYPE;
    case CAP_ENUM_DEF: {
        TSNode name = ts_node_child_by_field_name(node, "name", 4);
        return ts_node_is_null(name) ? SEC_DEF : SEC_TYPE;
    }
    case CAP_PREPROC_DEF:  return SEC_DEF;
    case CAP_PREPROC_FUNC: return SEC_DEF;
    case CAP_GLOBAL_VAR: {
        TSNode decl = ts_node_child_by_field_name(node, "declarator", 10);
        if (!ts_node_is_null(decl) && has_function_declarator(decl))
            return SEC_FUNCTIONS;
        return SEC_DATA;
    }
    case CAP_FUNC_DEF:     return SEC_FUNCTIONS;
    default: return SEC_DEF;
    }
}

const char *QUERY_SOURCE =
    "(function_definition) @func_def "
    "(struct_specifier) @struct_def "
    "(enum_specifier) @enum_def "
    "(type_definition) @typedef "
    "(declaration) @global_var "
    "(preproc_def) @preproc_def "
    "(preproc_function_def) @preproc_func "
    "(preproc_include) @include ";

struct file_entry *g_files;
int g_n_files, g_cap_files;
saha g_tags;

/* Tag accumulator — collects per-file tag names for cache storage */
char **g_file_tags;
int g_n_file_tags, g_cap_file_tags;

static archmap_cache *g_cache;

static int opt_follow_deps;
int opt_show_calls;
static char *opt_inc_paths[64];
static int opt_n_inc_paths;
static char *opt_defines[64];
static int opt_n_defines;
char *opt_strip_macros[64];
int opt_n_strip_macros;
size_t opt_strip_macro_lens[64];

enum { OPT_EXPAND = 256, OPT_COLLAPSE };

static struct option long_options[] = {
    {"deps",         no_argument,       NULL, 'd'},
    {"define",       required_argument, NULL, 'D'},
    {"include-dir",  required_argument, NULL, 'I'},
    {"calls",        no_argument,       NULL, 'c'},
    {"strip",        required_argument, NULL, 'S'},
    {"expand",       required_argument, NULL, OPT_EXPAND},
    {"collapse",     required_argument, NULL, OPT_COLLAPSE},
    {"help",         no_argument,       NULL, 'h'},
    {NULL, 0, NULL, 0}
};

static void print_usage(const char *prog) {
    const char *name = strrchr(prog, '/');
    name = name ? name + 1 : prog;
    fprintf(stderr,
        "Usage: %s [options] <path> [path ...]\n"
        "\n"
        "Options:\n"
        "  -d, --deps              Follow #include dependencies (BFS)\n"
        "  -D, --define MACRO      Define macro for preprocessor branch selection\n"
        "  -I, --include-dir DIR   Add include search path\n"
        "  -c, --calls             Show call graph edges under functions\n"
        "  -S, --strip MACRO       Strip project-specific qualifier macro from output\n"
        "  --expand GLOB           Force-expand collapsed files\n"
        "  --collapse GLOB         Collapse files (hide skeleton, show summary only)\n"
        "  -h, --help              Print this help and exit\n",
        name);
}

char *read_file_source(const char *path, long *out_length) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    if (length < 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *source = malloc((size_t)length + 1);
    if (!source) { fclose(f); return NULL; }
    size_t nread = fread(source, 1, (size_t)length, f);
    fclose(f);
    if ((long)nread != length) { free(source); return NULL; }
    source[length] = '\0';
    *out_length = length;
    return source;
}

int should_skip_fts_entry(FTSENT *ent) {
    if (ent->fts_name[0] == '.' && ent->fts_level > FTS_ROOTLEVEL) return 1;
    if (ent->fts_info == FTS_D && ent->fts_level > FTS_ROOTLEVEL &&
       (strcmp(ent->fts_name, "build") == 0 ||
        strcmp(ent->fts_name, "vendor") == 0 ||
        strcmp(ent->fts_name, "node_modules") == 0 ||
        strcmp(ent->fts_name, "third_party") == 0))
        return 1;
    return 0;
}

int is_source_file(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot) return 0;
    return (strcmp(dot, ".c") == 0 || strcmp(dot, ".h") == 0 ||
            strcmp(dot, ".cpp") == 0 || strcmp(dot, ".hpp") == 0 ||
            strcmp(dot, ".cc") == 0);
}

struct file_entry *add_file(const char *abs_path) {
    if (g_n_files >= g_cap_files) {
        g_cap_files = g_cap_files ? g_cap_files * 2 : 32;
        g_files = realloc(g_files, (size_t)g_cap_files * sizeof(*g_files));
    }
    struct file_entry *fe = &g_files[g_n_files++];
    memset(fe, 0, sizeof(*fe));
    fe->abs_path = strdup(abs_path);
    return fe;
}

struct symbol *add_symbol(struct file_entry *fe) {
    if (fe->n_syms >= fe->cap_syms) {
        fe->cap_syms = fe->cap_syms ? fe->cap_syms * 2 : 16;
        fe->syms = realloc(fe->syms, (size_t)fe->cap_syms * sizeof(*fe->syms));
    }
    struct symbol *s = &fe->syms[fe->n_syms++];
    memset(s, 0, sizeof(*s));
    return s;
}

static void add_callee(struct symbol *s, const char *name) {
    for (int i = 0; i < s->n_callees; i++)
        if (strcmp(s->callees[i], name) == 0) return;
    if (s->n_callees >= s->cap_callees) {
        s->cap_callees = s->cap_callees ? s->cap_callees * 2 : 8;
        s->callees = realloc(s->callees, (size_t)s->cap_callees * sizeof(char *));
    }
    s->callees[s->n_callees++] = strdup(name);
}

static int is_macro_defined(const char *name, char **defines, int n_defines) {
    for (int i = 0; i < n_defines; i++) {
        if (strcmp(defines[i], name) == 0) return 1;
    }
    return 0;
}

static int eval_ifdef(TSNode node, const char *source,
                      char **defines, int n_defines) {
    uint32_t start = ts_node_start_byte(node);
    const char *p = strstr(source + start, "ifndef");
    int is_ifndef = (p != NULL && (size_t)(p - (source + start)) < 20);

    TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
    if (ts_node_is_null(name_node)) return -1;

    uint32_t ns = ts_node_start_byte(name_node);
    uint32_t ne = ts_node_end_byte(name_node);
    uint32_t nlen = ne - ns;
    char *macro = malloc(nlen + 1);
    memcpy(macro, source + ns, nlen);
    macro[nlen] = '\0';

    /* __cplusplus guard: tree-sitter folds extern "C" { ... } into the
       #ifdef __cplusplus branch, swallowing all C declarations.  Pretend
       it's defined so the primary branch (containing the declarations)
       is emitted. */
    if (strcmp(macro, "__cplusplus") == 0) { free(macro); return is_ifndef ? 0 : 1; }

    int defined = is_macro_defined(macro, defines, n_defines);
    free(macro);

    if (is_ifndef) return defined ? 0 : 1;
    return defined ? 1 : 0;
}

static int eval_preproc_if(TSNode node, const char *source,
                           char **defines, int n_defines) {
    TSNode cond = ts_node_child_by_field_name(node, "condition", 9);
    if (ts_node_is_null(cond)) return -1;

    const char *ctype = ts_node_type(cond);

    if (strcmp(ctype, "preproc_defined") == 0) {
        uint32_t cc = ts_node_child_count(cond);
        for (uint32_t i = 0; i < cc; i++) {
            TSNode child = ts_node_child(cond, i);
            if (strcmp(ts_node_type(child), "identifier") == 0) {
                uint32_t s = ts_node_start_byte(child);
                uint32_t e = ts_node_end_byte(child);
                uint32_t len = e - s;
                char *macro = malloc(len + 1);
                memcpy(macro, source + s, len);
                macro[len] = '\0';
                int result = is_macro_defined(macro, defines, n_defines) ? 1 : 0;
                free(macro);
                return result;
            }
        }
        return -1;
    }

    if (strcmp(ctype, "unary_expression") == 0) {
        TSNode op = ts_node_child_by_field_name(cond, "operator", 8);
        if (ts_node_is_null(op)) return -1;
        uint32_t os = ts_node_start_byte(op);
        if (source[os] != '!') return -1;

        TSNode operand = ts_node_child_by_field_name(cond, "argument", 8);
        if (ts_node_is_null(operand)) return -1;
        if (strcmp(ts_node_type(operand), "preproc_defined") != 0) return -1;

        uint32_t cc = ts_node_child_count(operand);
        for (uint32_t i = 0; i < cc; i++) {
            TSNode child = ts_node_child(operand, i);
            if (strcmp(ts_node_type(child), "identifier") == 0) {
                uint32_t s = ts_node_start_byte(child);
                uint32_t e = ts_node_end_byte(child);
                uint32_t len = e - s;
                char *macro = malloc(len + 1);
                memcpy(macro, source + s, len);
                macro[len] = '\0';
                int result = is_macro_defined(macro, defines, n_defines) ? 0 : 1;
                free(macro);
                return result;
            }
        }
        return -1;
    }

    if (strcmp(ctype, "number_literal") == 0) {
        uint32_t s = ts_node_start_byte(cond);
        uint32_t e = ts_node_end_byte(cond);
        uint32_t len = e - s;
        if (len == 1 && source[s] == '0') return 0;
        if (len == 1 && source[s] == '1') return 1;
        return -1;
    }

    return -1;
}

static int should_skip_preproc(TSNode node, const char *source,
                               char **defines, int n_defines) {
    int in_alternative = 0;
    TSNode cur = ts_node_parent(node);

    while (!ts_node_is_null(cur)) {
        const char *type = ts_node_type(cur);

        if (strcmp(type, "preproc_else") == 0) {
            in_alternative = 1;
        } else if (strcmp(type, "preproc_elif") == 0) {
            int cond = eval_preproc_if(cur, source, defines, n_defines);
            if (in_alternative) {
                if (cond == 1) return 1;
            } else {
                if (cond != 1) return 1;
                in_alternative = 1;
            }
        } else if (strcmp(type, "preproc_elifdef") == 0) {
            int cond = eval_ifdef(cur, source, defines, n_defines);
            if (in_alternative) {
                if (cond == 1) return 1;
            } else {
                if (cond != 1) return 1;
                in_alternative = 1;
            }
        } else if (strcmp(type, "preproc_ifdef") == 0) {
            int cond = eval_ifdef(cur, source, defines, n_defines);
            if (cond == 1 && in_alternative) return 1;
            if (cond != 1 && !in_alternative) return 1;
            in_alternative = 0;
        } else if (strcmp(type, "preproc_if") == 0) {
            int cond = eval_preproc_if(cur, source, defines, n_defines);
            if (cond == 1 && in_alternative) return 1;
            if (cond != 1 && !in_alternative) return 1;
            in_alternative = 0;
        }

        cur = ts_node_parent(cur);
    }
    return 0;
}

static int is_file_scope(TSNode node) {
    TSNode cur = ts_node_parent(node);
    while (!ts_node_is_null(cur)) {
        const char *type = ts_node_type(cur);
        if (strcmp(type, "translation_unit") == 0)
            return 1;
        if (strcmp(type, "function_definition") == 0 ||
            strcmp(type, "compound_statement") == 0 ||
            strcmp(type, "field_declaration_list") == 0 ||
            strcmp(type, "enumerator_list") == 0)
            return 0;
        cur = ts_node_parent(cur);
    }
    return 0;
}

static char *resolve_include(const char *inc_path, int is_system,
                             const char *file_dir,
                             char **search_paths, int n_search) {
    char buf[PATH_MAX];
    if (!is_system && file_dir) {
        snprintf(buf, sizeof(buf), "%s/%s", file_dir, inc_path);
        char *real = realpath(buf, NULL);
        if (real) return real;
    }
    for (int i = 0; i < n_search; i++) {
        snprintf(buf, sizeof(buf), "%s/%s", search_paths[i], inc_path);
        char *real = realpath(buf, NULL);
        if (real) return real;
    }
    return NULL;
}


static char *extract_tag_name_src(TSNode node, const char *source) {
    TSNode name = ts_node_child_by_field_name(node, "name", 4);
    if (ts_node_is_null(name)) return NULL;
    const char *ntype = ts_node_type(name);
    if (strcmp(ntype, "type_identifier") != 0) return NULL;
    uint32_t s = ts_node_start_byte(name);
    uint32_t e = ts_node_end_byte(name);
    uint32_t len = e - s;
    char *tag = malloc(len + 1);
    memcpy(tag, source + s, len);
    tag[len] = '\0';
    return tag;
}

static int has_body(TSNode node) {
    TSNode body = ts_node_child_by_field_name(node, "body", 4);
    return !ts_node_is_null(body);
}

static void register_full_tags(TSNode node, const char *source) {
    const char *type = ts_node_type(node);
    if (strcmp(type, "struct_specifier") == 0 ||
        strcmp(type, "union_specifier") == 0 ||
        strcmp(type, "enum_specifier") == 0) {
        if (has_body(node)) {
            char *tag = extract_tag_name_src(node, source);
            if (tag) {
                saha_insert(&g_tags, tag, strlen(tag));
                /* Record for cache */
                if (g_n_file_tags >= g_cap_file_tags) {
                    g_cap_file_tags = g_cap_file_tags ? g_cap_file_tags * 2 : 8;
                    g_file_tags = realloc(g_file_tags,
                                          (size_t)g_cap_file_tags * sizeof(char *));
                }
                g_file_tags[g_n_file_tags++] = tag;  /* take ownership */
            }
        }
        return;
    }
    uint32_t cc = ts_node_child_count(node);
    for (uint32_t i = 0; i < cc; i++)
        register_full_tags(ts_node_child(node, i), source);
}

/* Check if a declaration node is a bare forward declaration like "struct X;"
   (no body, no declarator beyond the type itself) */
static int is_bare_forward_decl(TSNode node, const char *source,
                                uint32_t cap_idx, char **out_tag) {
    *out_tag = NULL;

    if (cap_idx == CAP_STRUCT_DEF || cap_idx == CAP_ENUM_DEF) {
        /* Standalone struct/enum specifier — forward if no body */
        if (!has_body(node)) {
            *out_tag = extract_tag_name_src(node, source);
            return *out_tag != NULL;
        }
        return 0;
    }

    if (cap_idx == CAP_GLOBAL_VAR) {
        /* declaration node — check if it's just "struct X;" with no declarator */
        TSNode type_node = ts_node_child_by_field_name(node, "type", 4);
        if (ts_node_is_null(type_node)) return 0;
        const char *ttype = ts_node_type(type_node);
        if (strcmp(ttype, "struct_specifier") != 0 &&
            strcmp(ttype, "union_specifier") != 0 &&
            strcmp(ttype, "enum_specifier") != 0)
            return 0;
        if (has_body(type_node)) return 0;

        /* Check there's no declarator */
        TSNode decl = ts_node_child_by_field_name(node, "declarator", 10);
        if (!ts_node_is_null(decl)) return 0;

        *out_tag = extract_tag_name_src(type_node, source);
        return *out_tag != NULL;
    }

    return 0;
}

void collect_callees(TSNode node, const char *source,
                     struct symbol *sym) {
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++) {
        TSNode child = ts_node_child(node, i);
        const char *type = ts_node_type(child);
        if (strcmp(type, "call_expression") == 0) {
            TSNode fn = ts_node_child_by_field_name(child, "function", 8);
            if (!ts_node_is_null(fn) &&
                strcmp(ts_node_type(fn), "identifier") == 0) {
                uint32_t s = ts_node_start_byte(fn);
                uint32_t e = ts_node_end_byte(fn);
                uint32_t len = e - s;
                char *name = malloc(len + 1);
                memcpy(name, source + s, len);
                name[len] = '\0';
                add_callee(sym, name);
                free(name);
            }
        }
        collect_callees(child, source, sym);
    }
}

void collect_file(const char *path, TSParser *parser, TSQuery *query,
                  char **defines, int n_defines,
                  char **search_paths, int n_search) {
    long length;
    char *source = read_file_source(path, &length);
    if (!source) return;

    TSTree *tree = ts_parser_parse_string(parser, NULL, source, (uint32_t)length);
    if (!tree) { free(source); return; }
    TSNode root = ts_tree_root_node(tree);

    TSQueryCursor *cursor = ts_query_cursor_new();
    ts_query_cursor_exec(cursor, query, root);

    struct file_entry *fe = add_file(path);

    TSQueryMatch match;
    while (ts_query_cursor_next_match(cursor, &match)) {
        if (match.capture_count < 1) continue;

        TSNode node = match.captures[0].node;
        uint32_t cap_idx = match.captures[0].index;

        /* Dedup: skip inner struct/enum when parent already captures */
        if (cap_idx == CAP_STRUCT_DEF || cap_idx == CAP_ENUM_DEF) {
            TSNode parent = ts_node_parent(node);
            if (!ts_node_is_null(parent)) {
                const char *ptype = ts_node_type(parent);
                if (strcmp(ptype, "type_definition") == 0 ||
                    strcmp(ptype, "declaration") == 0)
                    continue;
            }
        }

        if (!is_file_scope(node)) continue;

        if (should_skip_preproc(node, source, defines, n_defines))
            continue;

        /* Resolve #include paths and cache on file_entry */
        if (cap_idx == CAP_INCLUDE) {
            char *path_copy = strdup(path);
            char *file_dir = dirname(path_copy);
            uint32_t child_count = ts_node_child_count(node);
            for (uint32_t c = 0; c < child_count; c++) {
                TSNode child = ts_node_child(node, c);
                const char *type = ts_node_type(child);
                int is_system = 0;
                if (strcmp(type, "system_lib_string") == 0)
                    is_system = 1;
                else if (strcmp(type, "string_literal") == 0)
                    is_system = 0;
                else
                    continue;
                uint32_t s = ts_node_start_byte(child);
                uint32_t e = ts_node_end_byte(child);
                if (e - s < 3) continue;
                uint32_t inc_len = e - s - 2;
                char *inc_path = malloc(inc_len + 1);
                memcpy(inc_path, source + s + 1, inc_len);
                inc_path[inc_len] = '\0';
                char *resolved = resolve_include(inc_path, is_system, file_dir,
                                                 search_paths, n_search);
                free(inc_path);
                if (!resolved) continue;
                if (fe->n_includes % 16 == 0) {
                    fe->includes = realloc(fe->includes,
                        (size_t)(fe->n_includes + 16) * sizeof(char *));
                }
                fe->includes[fe->n_includes++] = resolved;
            }
            free(path_copy);
            continue;
        }

        /* Skip bare #define with no value (include guards) */
        if (cap_idx == CAP_PREPROC_DEF) {
            TSNode val = ts_node_child_by_field_name(node, "value", 5);
            if (ts_node_is_null(val)) continue;
        }

        /* Register full struct/enum/union tags for forward-decl dedup */
        if (cap_idx == CAP_STRUCT_DEF || cap_idx == CAP_ENUM_DEF ||
            cap_idx == CAP_TYPEDEF || cap_idx == CAP_GLOBAL_VAR) {
            register_full_tags(node, source);
        }

        struct symbol *sym = add_symbol(fe);
        sym->text = skeleton_text(source, node, cap_idx);
        sym->cap_idx = cap_idx;
        sym->start_line = ts_node_start_point(node).row + 1;
        sym->end_line = ts_node_end_point(node).row + 1;

        /* Skip incomplete declarations (e.g. CACHE_ALIGN on its own line) */
        if (cap_idx == CAP_GLOBAL_VAR) {
            uint32_t end_byte = ts_node_end_byte(node);
            uint32_t p = end_byte;
            while (p > ts_node_start_byte(node) &&
                   (source[p-1] == ' ' || source[p-1] == '\n' ||
                    source[p-1] == '\r' || source[p-1] == '\t'))
                p--;
            if (p == ts_node_start_byte(node) || source[p-1] != ';') {
                free(sym->text);
                fe->n_syms--;
                continue;
            }
        }

        sym->section = section_for_capture(cap_idx, node);

        char *fwd_tag = NULL;
        if (is_bare_forward_decl(node, source, cap_idx, &fwd_tag)) {
            sym->is_fwd_decl = 1;
            sym->tag_name = fwd_tag;
        }

        if (opt_show_calls && cap_idx == CAP_FUNC_DEF) {
            uint32_t child_count = ts_node_child_count(node);
            for (uint32_t c = 0; c < child_count; c++) {
                TSNode child = ts_node_child(node, c);
                if (strcmp(ts_node_type(child), "compound_statement") == 0) {
                    collect_callees(child, source, sym);
                    break;
                }
            }
        }
    }

    ts_query_cursor_delete(cursor);
    ts_tree_delete(tree);
    free(source);
}

static void collect_file_cached(const char *path, TSParser *parser,
                                TSQuery *query, char **defines,
                                int n_defines, char **search_paths,
                                int n_search) {
    if (g_cache) {
        struct file_entry tmp;
        char **tags; int n_tags;
        if (cache_lookup(g_cache, path, &tmp, &tags, &n_tags) == 1) {
            struct file_entry *fe = add_file(path);
            free(fe->abs_path);
            *fe = tmp;  /* transfer deep-copied data */
            for (int i = 0; i < n_tags; i++) {
                saha_insert(&g_tags, tags[i], strlen(tags[i]));
                free(tags[i]);
            }
            free(tags);
            return;  /* skip tree-sitter entirely */
        }
    }

    g_n_file_tags = 0;  /* reset tag accumulator */
    collect_file(path, parser, query, defines, n_defines,
                 search_paths, n_search);

    if (g_cache && g_n_files > 0) {
        cache_store(g_cache, path, &g_files[g_n_files - 1],
                    g_file_tags, g_n_file_tags);
    }
    for (int i = 0; i < g_n_file_tags; i++) free(g_file_tags[i]);
    g_n_file_tags = 0;
}

static void cleanup(void) {
    for (int i = 0; i < g_n_files; i++) {
        struct file_entry *fe = &g_files[i];
        free(fe->abs_path);
        free(fe->rel_path);
        for (int j = 0; j < fe->n_syms; j++) {
            free(fe->syms[j].text);
            free(fe->syms[j].tag_name);
            for (int k = 0; k < fe->syms[j].n_callees; k++)
                free(fe->syms[j].callees[k]);
            free(fe->syms[j].callees);
        }
        free(fe->syms);
        for (int j = 0; j < fe->n_includes; j++)
            free(fe->includes[j]);
        free(fe->includes);
    }
    free(g_files);

    output_cleanup();

    saha_destroy(&g_tags);
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "perf") == 0) {
        argv[1] = argv[0];
        return perf_main(argc - 1, argv + 1);
    }

    int opt;
    while ((opt = getopt_long(argc, argv, "dD:I:cS:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'd': opt_follow_deps = 1; break;
        case 'D':
            if (opt_n_defines < 64) opt_defines[opt_n_defines++] = optarg;
            break;
        case 'I':
            if (opt_n_inc_paths < 64) opt_inc_paths[opt_n_inc_paths++] = optarg;
            break;
        case 'c': opt_show_calls = 1; break;
        case 'S':
            if (opt_n_strip_macros < 64) {
                opt_strip_macros[opt_n_strip_macros] = optarg;
                opt_strip_macro_lens[opt_n_strip_macros] = strlen(optarg);
                opt_n_strip_macros++;
            }
            break;
        case OPT_EXPAND:
            if (opt_n_expand < 64) opt_expand_globs[opt_n_expand++] = optarg;
            break;
        case OPT_COLLAPSE:
            if (opt_n_collapse < 64) opt_collapse_globs[opt_n_collapse++] = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }
    if (optind >= argc) {
        print_usage(argv[0]);
        return 1;
    }

    saha_init(&g_tags);

    char *opts_str = cache_make_opts_str(opt_show_calls,
        opt_defines, opt_n_defines, opt_strip_macros, opt_n_strip_macros,
        opt_inc_paths, opt_n_inc_paths);
    g_cache = cache_open(".", opts_str);
    free(opts_str);

    TSParser *parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_c());

    uint32_t err_offset;
    TSQueryError err_type;
    TSQuery *query = ts_query_new(tree_sitter_c(), QUERY_SOURCE, strlen(QUERY_SOURCE),
                                  &err_offset, &err_type);
    if (!query) {
        fprintf(stderr, "Invalid query at offset %u (error type %d)\n",
                err_offset, err_type);
        ts_parser_delete(parser);
        return 1;
    }

    if (opt_follow_deps) {
        /* BFS dependency walk */
        int vis_cap = 64, vis_count = 0, queue_head = 0;
        char **visited = malloc((size_t)vis_cap * sizeof(char *));

        FTS *ftsp = fts_open(argv + optind, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
        if (!ftsp) {
            perror("fts_open");
            free(visited);
            ts_query_delete(query);
            ts_parser_delete(parser);
            return 1;
        }
        FTSENT *ent;
        while ((ent = fts_read(ftsp))) {
            if (should_skip_fts_entry(ent)) {
                fts_set(ftsp, ent, FTS_SKIP);
                continue;
            }
            if (ent->fts_info == FTS_F && is_source_file(ent->fts_name)) {
                char *real = realpath(ent->fts_path, NULL);
                if (real) {
                    visited_add(&visited, &vis_count, &vis_cap, real);
                    free(real);
                }
            }
        }
        fts_close(ftsp);

        /* BFS: collect files and discover new includes */
        while (queue_head < vis_count) {
            const char *file = visited[queue_head++];
            collect_file_cached(file, parser, query, opt_defines, opt_n_defines,
                                opt_inc_paths, opt_n_inc_paths);

            /* Read cached includes from the just-collected file_entry */
            struct file_entry *fe = &g_files[g_n_files - 1];
            for (int i = 0; i < fe->n_includes; i++)
                visited_add(&visited, &vis_count, &vis_cap, fe->includes[i]);
        }

        for (int i = 0; i < vis_count; i++) free(visited[i]);
        free(visited);
    } else {
        /* Walk paths via fts */
        FTS *ftsp = fts_open(argv + optind, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
        if (!ftsp) {
            perror("fts_open");
            ts_query_delete(query);
            ts_parser_delete(parser);
            return 1;
        }

        FTSENT *ent;
        while ((ent = fts_read(ftsp))) {
            if (should_skip_fts_entry(ent)) {
                fts_set(ftsp, ent, FTS_SKIP);
                continue;
            }

            if (ent->fts_info == FTS_F && is_source_file(ent->fts_name)) {
                collect_file_cached(ent->fts_path, parser, query,
                                    opt_defines, opt_n_defines,
                                    opt_inc_paths, opt_n_inc_paths);
            }
        }

        fts_close(ftsp);
    }

    if (g_cache) { cache_close(g_cache); g_cache = NULL; }

    print_tree();
    cleanup();

    ts_query_delete(query);
    ts_parser_delete(parser);
    return 0;
}
