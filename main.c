/* archmap.c - Generates a semantic skeleton of a C/C++ codebase */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fts.h>
#include <getopt.h>
#include <libgen.h>
#include <limits.h>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-c.h>

/* Verstable string set for struct/enum/union tag deduplication */
#define NAME tag_set
#define KEY_TY char *
#define HASH_FN vt_hash_string
#define CMPR_FN vt_cmpr_string
#include "verstable.h"

/* Capture index names — must match order in QUERY_SOURCE */
enum {
    CAP_FUNC_DEF = 0,
    CAP_STRUCT_DEF,
    CAP_ENUM_DEF,
    CAP_TYPEDEF,
    CAP_GLOBAL_VAR,
    CAP_PREPROC_DEF,
    CAP_PREPROC_FUNC,
    CAP_INCLUDE,
};

const char *QUERY_SOURCE =
    "(function_definition) @func_def "
    "(struct_specifier) @struct_def "
    "(enum_specifier) @enum_def "
    "(type_definition) @typedef "
    "(declaration) @global_var "
    "(preproc_def) @preproc_def "
    "(preproc_function_def) @preproc_func "
    "(preproc_include) @include ";

/* ── Data structures for buffered output ───────────────────────── */

struct symbol {
    char *text;
    char **callees;
    int n_callees;
    int cap_callees;
    int is_fwd_decl;
    char *tag_name;
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

static struct file_entry *g_files;
static int g_n_files, g_cap_files;
static tag_set g_tags;

/* ── CLI options ───────────────────────────────────────────────── */

static int opt_follow_deps;
static int opt_show_calls;
static char *opt_inc_paths[64];
static int opt_n_inc_paths;
static char *opt_defines[64];
static int opt_n_defines;
static char *opt_strip_macros[64];
static int opt_n_strip_macros;
static size_t opt_strip_macro_lens[64];

static struct option long_options[] = {
    {"deps",         no_argument,       NULL, 'd'},
    {"define",       required_argument, NULL, 'D'},
    {"include-dir",  required_argument, NULL, 'I'},
    {"calls",        no_argument,       NULL, 'c'},
    {"strip",        required_argument, NULL, 'S'},
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
        "  -h, --help              Print this help and exit\n",
        name);
}

/* ── Helpers ───────────────────────────────────────────────────── */

static char *read_file_source(const char *path, long *out_length) {
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

static int should_skip_fts_entry(FTSENT *ent) {
    if (ent->fts_name[0] == '.') return 1;
    if (ent->fts_info == FTS_D && ent->fts_level > FTS_ROOTLEVEL &&
       (strcmp(ent->fts_name, "build") == 0 ||
        strcmp(ent->fts_name, "vendor") == 0 ||
        strcmp(ent->fts_name, "node_modules") == 0 ||
        strcmp(ent->fts_name, "third_party") == 0))
        return 1;
    return 0;
}

static int is_source_file(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot) return 0;
    return (strcmp(dot, ".c") == 0 || strcmp(dot, ".h") == 0 ||
            strcmp(dot, ".cpp") == 0 || strcmp(dot, ".hpp") == 0 ||
            strcmp(dot, ".cc") == 0);
}

static struct file_entry *add_file(const char *abs_path) {
    if (g_n_files >= g_cap_files) {
        g_cap_files = g_cap_files ? g_cap_files * 2 : 32;
        g_files = realloc(g_files, (size_t)g_cap_files * sizeof(*g_files));
    }
    struct file_entry *fe = &g_files[g_n_files++];
    memset(fe, 0, sizeof(*fe));
    fe->abs_path = strdup(abs_path);
    return fe;
}

static struct symbol *add_symbol(struct file_entry *fe) {
    if (fe->n_syms >= fe->cap_syms) {
        fe->cap_syms = fe->cap_syms ? fe->cap_syms * 2 : 16;
        fe->syms = realloc(fe->syms, (size_t)fe->cap_syms * sizeof(*fe->syms));
    }
    struct symbol *s = &fe->syms[fe->n_syms++];
    memset(s, 0, sizeof(*s));
    return s;
}

static void add_callee(struct symbol *s, const char *name) {
    /* deduplicate */
    for (int i = 0; i < s->n_callees; i++)
        if (strcmp(s->callees[i], name) == 0) return;
    if (s->n_callees >= s->cap_callees) {
        s->cap_callees = s->cap_callees ? s->cap_callees * 2 : 8;
        s->callees = realloc(s->callees, (size_t)s->cap_callees * sizeof(char *));
    }
    s->callees[s->n_callees++] = strdup(name);
}

/* ── Skeleton text generation (to malloc'd string via open_memstream) ── */

/* Strip storage-class specifiers and inline — everything in the skeleton
   is file-scope by definition, so these waste tokens. */
static void strip_qualifiers(char *text) {
    char *src = text;
    for (;;) {
        if (strncmp(src, "static ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "inline ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "extern ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "_Noreturn ", 10) == 0) { src += 10; continue; }
        /* __attribute__((...)) — find matching )) */
        if (strncmp(src, "__attribute__((", 15) == 0) {
            int depth = 2;
            char *p = src + 15;
            while (*p && depth > 0) {
                if (*p == '(') depth++;
                else if (*p == ')') depth--;
                p++;
            }
            while (*p == ' ' || *p == '\n') p++;
            src = p;
            continue;
        }
        /* User-specified -S macros */
        int matched = 0;
        for (int i = 0; i < opt_n_strip_macros; i++) {
            if (strncmp(src, opt_strip_macros[i], opt_strip_macro_lens[i]) == 0 &&
                (src[opt_strip_macro_lens[i]] == ' ' ||
                 src[opt_strip_macro_lens[i]] == '\n' ||
                 src[opt_strip_macro_lens[i]] == '\t')) {
                src += opt_strip_macro_lens[i];
                while (*src == ' ' || *src == '\n' || *src == '\t') src++;
                matched = 1;
                break;
            }
        }
        if (matched) continue;
        break;
    }
    if (src != text)
        memmove(text, src, strlen(src) + 1);
}

static void strip_trailing_attribute(char *text) {
    char *last = NULL;
    char *p = text;
    while ((p = strstr(p, "__attribute__((")) != NULL) {
        last = p;
        p++;
    }
    if (!last) return;
    /* Verify it's after a closing brace/paren (trailing position) */
    char *before = last - 1;
    while (before >= text && (*before == ' ' || *before == '\n')) before--;
    if (before < text || (*before != ')' && *before != '}')) return;
    /* Find matching )) */
    int depth = 2;
    char *end = last + 15;
    while (*end && depth > 0) {
        if (*end == '(') depth++;
        else if (*end == ')') depth--;
        end++;
    }
    /* Remove: shift everything after the attribute back */
    while (*end == ' ') end++;
    memmove(last, end, strlen(end) + 1);
}

static void strip_comments(char *text) {
    char *r = text, *w = text;
    while (*r) {
        /* Skip string literals verbatim */
        if (*r == '"' || *r == '\'') {
            char q = *r;
            *w++ = *r++;
            while (*r && *r != q) {
                if (*r == '\\' && r[1]) *w++ = *r++;
                *w++ = *r++;
            }
            if (*r) *w++ = *r++;
            continue;
        }
        /* Line comment — skip to newline, keep the newline */
        if (r[0] == '/' && r[1] == '/') {
            while (*r && *r != '\n') r++;
            continue;
        }
        /* Block comment — skip entirely */
        if (r[0] == '/' && r[1] == '*') {
            r += 2;
            while (*r && !(r[0] == '*' && r[1] == '/')) r++;
            if (*r) r += 2;
            continue;
        }
        *w++ = *r++;
    }
    *w = '\0';
}

static void normalize_whitespace(char *text) {
    char *r = text, *w = text;
    while (*r) {
        char *line_start = w;
        /* Copy leading whitespace (indentation) as-is */
        while (*r == ' ' || *r == '\t') *w++ = *r++;
        /* Process content: collapse interior space runs */
        while (*r && *r != '\n') {
            if (*r == '"' || *r == '\'') {
                char q = *r;
                *w++ = *r++;
                while (*r && *r != q) {
                    if (*r == '\\' && r[1]) *w++ = *r++;
                    *w++ = *r++;
                }
                if (*r) *w++ = *r++;
                continue;
            }
            if (*r == ' ' || *r == '\t') {
                *w++ = ' ';
                while (*r == ' ' || *r == '\t') r++;
                continue;
            }
            *w++ = *r++;
        }
        /* Trim trailing spaces before newline */
        while (w > line_start && (w[-1] == ' ' || w[-1] == '\t')) w--;
        /* Copy newline */
        if (*r == '\n') *w++ = *r++;
    }
    *w = '\0';
}

static void join_continuation_lines(char *text) {
    char *r = text, *w = text;
    while (*r) {
        char *line_start = w;
        int is_preproc = 0;

        /* Copy first segment of line */
        while (*r && *r != '\n') *w++ = *r++;

        /* Check if original line starts with # */
        for (char *p = line_start; p < w; p++) {
            if (*p == ' ' || *p == '\t') continue;
            if (*p == '#') is_preproc = 1;
            break;
        }

        /* Inner loop: keep joining until line is complete */
        while (*r == '\n' && !is_preproc) {
            /* Find last non-whitespace char of current line */
            char *ce = w;
            while (ce > line_start && (ce[-1] == ' ' || ce[-1] == '\t')) ce--;
            char last = (ce > line_start) ? ce[-1] : '\0';

            /* Rule 1: dangling semicolon on next line */
            char *nc = r + 1;
            while (*nc == ' ' || *nc == '\t') nc++;
            if (*nc == ';' && (nc[1] == '\n' || nc[1] == '\0')) {
                *w++ = ';';
                r = nc + 1;
                break;
            }

            /* Rule 2: line complete? */
            if (last == ';' || last == ',' || last == '{' ||
                last == '}' || last == '\\' || last == ')' ||
                last == '\0')
                break;

            /* Incomplete — join next line */
            r++;                                      /* skip \n */
            while (*r == ' ' || *r == '\t') r++;      /* skip indent */
            *w++ = ' ';                               /* single space */
            while (*r && *r != '\n') *w++ = *r++;     /* copy next line */
        }

        if (*r == '\n') *w++ = *r++;
    }
    *w = '\0';
}

static void collapse_blank_lines(char *text) {
    char *r = text, *w = text;
    while (*r) {
        *w++ = *r++;
        if (r[-1] == '\n') {
            while (*r == '\n') r++;
        }
    }
    *w = '\0';
}

static void postprocess_skeleton(char *buf, int do_strip) {
    if (do_strip) {
        strip_qualifiers(buf);
        strip_trailing_attribute(buf);
    }
    strip_comments(buf);
    normalize_whitespace(buf);
    join_continuation_lines(buf);
    collapse_blank_lines(buf);
}

static void fprint_node_text(FILE *out, const char *source,
                             uint32_t start, uint32_t end, int add_semi) {
    uint32_t len = end - start;
    int has_nl = (len > 0 && source[start + len - 1] == '\n');
    if (add_semi)
        fprintf(out, "%.*s;%s", len, source + start, has_nl ? "" : "\n");
    else
        fprintf(out, "%.*s%s", len, source + start, has_nl ? "" : "\n");
}

static int is_simple_sequential_enum(TSNode enum_node, int threshold) {
    TSNode body = ts_node_child_by_field_name(enum_node, "body", 4);
    if (ts_node_is_null(body)) return 0;
    int n = 0, n_valued = 0, first_valued = 0;
    uint32_t cc = ts_node_child_count(body);
    for (uint32_t i = 0; i < cc; i++) {
        TSNode child = ts_node_child(body, i);
        if (strcmp(ts_node_type(child), "enumerator") != 0) continue;
        n++;
        TSNode val = ts_node_child_by_field_name(child, "value", 5);
        if (!ts_node_is_null(val)) {
            n_valued++;
            if (n == 1) first_valued = 1;
        }
    }
    if (n <= threshold) return 0;
    return (n_valued == 0 || (n_valued == 1 && first_valued));
}

static void fprint_compact_enum(FILE *out, TSNode full_node,
                                TSNode enum_node, const char *source) {
    /* Print everything from full_node start to enum body '{' */
    uint32_t node_start = ts_node_start_byte(full_node);
    TSNode body = ts_node_child_by_field_name(enum_node, "body", 4);
    uint32_t body_start = ts_node_start_byte(body);
    fprintf(out, "%.*s{ ", (int)(body_start - node_start), source + node_start);

    /* Print enumerators comma-separated */
    uint32_t cc = ts_node_child_count(body);
    int first = 1;
    for (uint32_t i = 0; i < cc; i++) {
        TSNode child = ts_node_child(body, i);
        if (strcmp(ts_node_type(child), "enumerator") != 0) continue;
        if (!first) fprintf(out, ", ");
        first = 0;
        uint32_t cs = ts_node_start_byte(child);
        uint32_t ce = ts_node_end_byte(child);
        fprintf(out, "%.*s", (int)(ce - cs), source + cs);
    }

    /* Print everything from enum body '}' to full_node end */
    uint32_t body_end = ts_node_end_byte(body);
    uint32_t node_end = ts_node_end_byte(full_node);
    fprintf(out, " }%.*s\n", (int)(node_end - body_end), source + body_end);
}

static char *skeleton_text(const char *source, TSNode node, uint32_t cap_idx) {
    char *buf = NULL;
    size_t buf_len = 0;
    FILE *mem = open_memstream(&buf, &buf_len);
    if (!mem) return NULL;

    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);

    if (cap_idx == CAP_FUNC_DEF) {
        uint32_t child_count = ts_node_child_count(node);
        for (uint32_t c = 0; c < child_count; c++) {
            TSNode child = ts_node_child(node, c);
            if (strcmp(ts_node_type(child), "compound_statement") == 0) {
                uint32_t body_start = ts_node_start_byte(child);
                /* Trim trailing whitespace before the body */
                uint32_t trim_end = body_start;
                while (trim_end > start &&
                       (source[trim_end-1] == ' ' || source[trim_end-1] == '\n' ||
                        source[trim_end-1] == '\r' || source[trim_end-1] == '\t'))
                    trim_end--;
                fprint_node_text(mem, source, start, trim_end, 1);
                fclose(mem);
                postprocess_skeleton(buf, 1);
                return buf;
            }
        }
    }

    if (cap_idx == CAP_GLOBAL_VAR) {
        /* Truncate large initializer lists and string literals */
        uint32_t cc = ts_node_child_count(node);
        for (uint32_t c = 0; c < cc; c++) {
            TSNode child = ts_node_child(node, c);
            if (strcmp(ts_node_type(child), "init_declarator") == 0) {
                TSNode val = ts_node_child_by_field_name(child, "value", 5);
                if (!ts_node_is_null(val)) {
                    const char *vtype = ts_node_type(val);
                    const char *placeholder = NULL;
                    if (strcmp(vtype, "initializer_list") == 0)
                        placeholder = "{ ... };\n";
                    else if (strcmp(vtype, "string_literal") == 0 ||
                             strcmp(vtype, "concatenated_string") == 0)
                        placeholder = "\"...\";\n";
                    if (placeholder) {
                        uint32_t val_start = ts_node_start_byte(val);
                        fprint_node_text(mem, source, start, val_start, 0);
                        /* Trim trailing whitespace so "=\n  " becomes "= " */
                        long pos = ftell(mem);
                        while (pos > 0) {
                            fseek(mem, pos - 1, SEEK_SET);
                            int ch = fgetc(mem);
                            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
                                pos--;
                            else
                                break;
                        }
                        fseek(mem, pos, SEEK_SET);
                        fprintf(mem, " %s", placeholder);
                        fclose(mem);
                        postprocess_skeleton(buf, 1);
                        return buf;
                    }
                }
            }
        }
        /* Fall through to default handling */
    }

    if (cap_idx == CAP_STRUCT_DEF || cap_idx == CAP_ENUM_DEF) {
        if (cap_idx == CAP_ENUM_DEF &&
            is_simple_sequential_enum(node, 6)) {
            fprint_compact_enum(mem, node, node, source);
            fclose(mem);
            /* Compact output ends with "}\n"; insert semicolon */
            size_t len = strlen(buf);
            if (len > 0 && buf[len-1] == '\n') {
                buf = realloc(buf, len + 2);
                buf[len-1] = ';';
                buf[len] = '\n';
                buf[len+1] = '\0';
            }
            strip_qualifiers(buf);
            return buf;
        }
        fprint_node_text(mem, source, start, end, 1);
        fclose(mem);
        postprocess_skeleton(buf, 1);
        return buf;
    }

    /* Compact typedef-wrapped sequential enums */
    if (cap_idx == CAP_TYPEDEF) {
        uint32_t cc = ts_node_child_count(node);
        for (uint32_t c = 0; c < cc; c++) {
            TSNode child = ts_node_child(node, c);
            if (strcmp(ts_node_type(child), "enum_specifier") == 0 &&
                is_simple_sequential_enum(child, 6)) {
                fprint_compact_enum(mem, node, child, source);
                fclose(mem);
                strip_qualifiers(buf);
                return buf;
            }
        }
    }

    int strip = (cap_idx == CAP_GLOBAL_VAR || cap_idx == CAP_TYPEDEF);
    fprint_node_text(mem, source, start, end, 0);
    fclose(mem);
    postprocess_skeleton(buf, strip);
    return buf;
}

/* ── Preprocessor evaluation (unchanged logic) ─────────────────── */

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

/* ── Include resolution ────────────────────────────────────────── */

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


/* ── Tag name extraction ───────────────────────────────────────── */

/* Extract tag name given access to source text */
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

/* Check if a node has a body (field_declaration_list for struct/union,
   enumerator_list for enum) */
static int has_body(TSNode node) {
    TSNode body = ts_node_child_by_field_name(node, "body", 4);
    return !ts_node_is_null(body);
}

/* Search inside a node for struct/union/enum specifiers and extract their tags
   into g_tags if they have bodies. Also look inside type_definition children. */
static void register_full_tags(TSNode node, const char *source) {
    const char *type = ts_node_type(node);
    if (strcmp(type, "struct_specifier") == 0 ||
        strcmp(type, "union_specifier") == 0 ||
        strcmp(type, "enum_specifier") == 0) {
        if (has_body(node)) {
            char *tag = extract_tag_name_src(node, source);
            if (tag) {
                tag_set_itr itr = tag_set_get(&g_tags, tag);
                if (tag_set_is_end(itr)) {
                    tag_set_insert(&g_tags, tag);
                } else {
                    free(tag);
                }
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

/* ── Call graph extraction ─────────────────────────────────────── */

static void collect_callees(TSNode node, const char *source,
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

/* ── collect_file: parse, query, filter, buffer symbols ────────── */

static void collect_file(const char *path, TSParser *parser, TSQuery *query,
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

        /* Only file-scope nodes */
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

        /* Detect forward declarations */
        char *fwd_tag = NULL;
        if (is_bare_forward_decl(node, source, cap_idx, &fwd_tag)) {
            sym->is_fwd_decl = 1;
            sym->tag_name = fwd_tag;
        }

        /* Collect call graph edges */
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

/* ── Visited set ───────────────────────────────────────────────── */

static int visited_add(char ***visited, int *count, int *cap, const char *path) {
    for (int i = 0; i < *count; i++) {
        if (strcmp((*visited)[i], path) == 0) return 0;
    }
    if (*count >= *cap) {
        *cap *= 2;
        *visited = realloc(*visited, (size_t)*cap * sizeof(char *));
    }
    (*visited)[(*count)++] = strdup(path);
    return 1;
}

/* ── Common prefix computation ─────────────────────────────────── */

static void compute_common_prefix(char *out, size_t out_sz) {
    if (g_n_files == 0) { out[0] = '\0'; return; }

    /* Start with the first file's directory */
    char *first = g_files[0].abs_path;
    char *last_slash = strrchr(first, '/');
    size_t pfx_len = last_slash ? (size_t)(last_slash - first + 1) : 0;
    if (pfx_len >= out_sz) pfx_len = out_sz - 1;
    memcpy(out, first, pfx_len);
    out[pfx_len] = '\0';

    for (int i = 1; i < g_n_files; i++) {
        const char *p = g_files[i].abs_path;
        size_t j = 0;
        while (j < pfx_len && p[j] == out[j]) j++;
        /* Back up to last '/' */
        while (j > 0 && out[j - 1] != '/') j--;
        pfx_len = j;
        out[pfx_len] = '\0';
    }
}

/* ── Topological sort by include order ──────────────────────────── */

static void topo_sort_files(void) {
    int n = g_n_files;
    if (n <= 1) return;

    /* Build path→index map using linear search (small n) */
    int *in_degree = calloc((size_t)n, sizeof(int));
    /* Adjacency: adj[i] lists indices that depend on i (i must come before them) */
    int *adj = calloc((size_t)n * (size_t)n, sizeof(int));  /* adj[i*n + j] = 1 */

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < g_files[i].n_includes; k++) {
            /* Find which file index this resolves to */
            for (int j = 0; j < n; j++) {
                if (j != i && strcmp(g_files[j].abs_path, g_files[i].includes[k]) == 0) {
                    /* i includes j → j must come before i */
                    if (!adj[j * n + i]) {
                        adj[j * n + i] = 1;
                        in_degree[i]++;
                    }
                    break;
                }
            }
        }
    }

    /* Kahn's algorithm */
    int *queue = malloc((size_t)n * sizeof(int));
    int head = 0, tail = 0;
    for (int i = 0; i < n; i++) {
        if (in_degree[i] == 0)
            queue[tail++] = i;
    }

    int *order = malloc((size_t)n * sizeof(int));
    int count = 0;
    while (head < tail) {
        int u = queue[head++];
        order[count++] = u;
        for (int v = 0; v < n; v++) {
            if (adj[u * n + v]) {
                in_degree[v]--;
                if (in_degree[v] == 0)
                    queue[tail++] = v;
            }
        }
    }
    /* Any remaining nodes (cycles) — append in original order */
    for (int i = 0; i < n; i++) {
        if (in_degree[i] > 0)
            order[count++] = i;
    }

    /* Reorder g_files in-place via a temp copy */
    struct file_entry *tmp = malloc((size_t)n * sizeof(*tmp));
    for (int i = 0; i < n; i++)
        tmp[i] = g_files[order[i]];
    memcpy(g_files, tmp, (size_t)n * sizeof(*tmp));

    free(tmp);
    free(order);
    free(queue);
    free(adj);
    free(in_degree);
}

/* ── Tree printer ──────────────────────────────────────────────── */

static void print_indented(const char *text, int indent) {
    if (!text) return;
    const char *p = text;
    while (*p) {
        const char *nl = strchr(p, '\n');
        if (!nl) nl = p + strlen(p);
        printf("%*s%.*s\n", indent, "", (int)(nl - p), p);
        if (*nl == '\n') nl++;
        p = nl;
        if (*p == '\0') break;
    }
}

static void print_tree(void) {
    /* Topological sort by include dependencies */
    topo_sort_files();

    /* Compute common prefix and relative paths */
    char prefix[PATH_MAX];
    compute_common_prefix(prefix, sizeof(prefix));
    size_t pfx_len = strlen(prefix);

    for (int i = 0; i < g_n_files; i++) {
        const char *ap = g_files[i].abs_path;
        if (strncmp(ap, prefix, pfx_len) == 0)
            g_files[i].rel_path = strdup(ap + pfx_len);
        else
            g_files[i].rel_path = strdup(ap);
    }

    char prev_dir[PATH_MAX] = "";

    for (int i = 0; i < g_n_files; i++) {
        struct file_entry *fe = &g_files[i];
        if (fe->n_syms == 0) continue;

        /* Extract directory portion */
        char dir[PATH_MAX] = "";
        const char *base = fe->rel_path;
        const char *last_slash = strrchr(fe->rel_path, '/');
        if (last_slash) {
            size_t dlen = (size_t)(last_slash - fe->rel_path);
            if (dlen >= sizeof(dir)) dlen = sizeof(dir) - 1;
            memcpy(dir, fe->rel_path, dlen);
            dir[dlen] = '\0';
            base = last_slash + 1;
        }

        /* Print directory header if changed */
        if (strcmp(dir, prev_dir) != 0) {
            if (dir[0] != '\0')
                printf("%s/\n", dir);
            snprintf(prev_dir, sizeof(prev_dir), "%s", dir);
        }

        /* File name, indented 2 spaces */
        printf("  %s\n", base);

        /* Symbols, indented 4 spaces */
        for (int j = 0; j < fe->n_syms; j++) {
            struct symbol *s = &fe->syms[j];

            /* Suppress forward declarations whose tag has a full definition */
            if (s->is_fwd_decl && s->tag_name) {
                tag_set_itr itr = tag_set_get(&g_tags, s->tag_name);
                if (!tag_set_is_end(itr))
                    continue;
            }

            print_indented(s->text, 4);

            /* Call edges, indented 6 spaces */
            if (opt_show_calls && s->n_callees > 0) {
                printf("      \xe2\x86\x92 ");
                for (int k = 0; k < s->n_callees; k++) {
                    printf("%s%s", s->callees[k],
                           k < s->n_callees - 1 ? ", " : "\n");
                }
            }
        }
    }
}

/* ── Cleanup ───────────────────────────────────────────────────── */

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

    /* Clean up tag set — free the stored keys */
    for (tag_set_itr itr = tag_set_first(&g_tags);
         !tag_set_is_end(itr);
         itr = tag_set_next(itr)) {
        free(itr.data->key);
    }
    tag_set_cleanup(&g_tags);
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
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

    tag_set_init(&g_tags);

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
            collect_file(file, parser, query, opt_defines, opt_n_defines,
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
                collect_file(ent->fts_path, parser, query,
                             opt_defines, opt_n_defines,
                             opt_inc_paths, opt_n_inc_paths);
            }
        }

        fts_close(ftsp);
    }

    /* Print the collected tree */
    print_tree();
    cleanup();

    ts_query_delete(query);
    ts_parser_delete(parser);
    return 0;
}
