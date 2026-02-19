/* skeleton.c — Skeleton text generation and postprocessing */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "skeleton.h"

/* Globals from main.c */
extern char *opt_strip_macros[64];
extern int opt_n_strip_macros;
extern size_t opt_strip_macro_lens[64];

/* Strip storage-class specifiers and inline — everything in the skeleton
   is file-scope by definition, so these waste tokens. */
static void strip_qualifiers(char *text) {
    char *src = text;
    for (;;) {
        if (strncmp(src, "static ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "inline ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "extern ", 7) == 0) { src += 7; continue; }
        if (strncmp(src, "const ", 6) == 0) { src += 6; continue; }
        if (strncmp(src, "volatile ", 9) == 0) { src += 9; continue; }
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

void strip_struct_refs(char *text) {
    char *r = text, *w = text;
    while (*r) {
        /* Preserve string literal contents */
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
        if (strncmp(r, "struct ", 7) == 0) { r += 7; continue; }
        if (strncmp(r, "union ", 6) == 0) { r += 6; continue; }
        *w++ = *r++;
    }
    *w = '\0';
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

static void collapse_to_single_line(char *text) {
    char *r = text, *w = text;
    while (*r) {
        /* Preserve string literal contents */
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
        if (*r == '\n' || *r == '\r') {
            /* Replace newline + surrounding whitespace with single space */
            while (w > text && (w[-1] == ' ' || w[-1] == '\t')) w--;
            r++;
            while (*r == ' ' || *r == '\t' || *r == '\n' || *r == '\r') r++;
            if (*r) *w++ = ' ';
            continue;
        }
        *w++ = *r++;
    }
    /* Trim trailing whitespace */
    while (w > text && (w[-1] == ' ' || w[-1] == '\t')) w--;
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
    collapse_to_single_line(buf);
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

char *skeleton_text(const char *source, TSNode node, uint32_t cap_idx) {
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
    }

    if (cap_idx == CAP_STRUCT_DEF || cap_idx == CAP_ENUM_DEF) {
        if (cap_idx == CAP_ENUM_DEF &&
            is_simple_sequential_enum(node, 0)) {
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
            collapse_to_single_line(buf);
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
                is_simple_sequential_enum(child, 0)) {
                fprint_compact_enum(mem, node, child, source);
                fclose(mem);
                strip_qualifiers(buf);
                collapse_to_single_line(buf);
                return buf;
            }
        }
    }

    if (cap_idx == CAP_PREPROC_FUNC) {
        TSNode params = ts_node_child_by_field_name(node, "parameters", 10);
        if (!ts_node_is_null(params)) {
            uint32_t params_end = ts_node_end_byte(params);
            fprint_node_text(mem, source, start, params_end, 0);
            fclose(mem);
            postprocess_skeleton(buf, 0);
            return buf;
        }
    }

    int strip = (cap_idx == CAP_GLOBAL_VAR || cap_idx == CAP_TYPEDEF);
    fprint_node_text(mem, source, start, end, 0);
    fclose(mem);
    postprocess_skeleton(buf, strip);
    return buf;
}
