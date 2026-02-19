/* output.c — Output formatting, topo sort, sidecar loading */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fnmatch.h>
#include <limits.h>

#include "output.h"
#include "skeleton.h"
#include "perf_analysis.h"
#include "hmap/hmap_avx512.h"

/* Globals from main.c */
extern struct file_entry *g_files;
extern int g_n_files;
extern int opt_show_calls;
extern saha g_tags;

static const char *section_headers[SEC_COUNT] = {
    "[struct]", "[type]", "[def]", "[data]", "[functions]"
};

/* ── Sidecar data (owned by this module) ──────────────────────────────── */

struct sidecar_entry {
    char *glob;
    char *summary;
    int collapse;
};

static struct sidecar_entry *g_sidecar;
static int g_n_sidecar, g_cap_sidecar;

char *opt_expand_globs[64];
int opt_n_expand;
char *opt_collapse_globs[64];
int opt_n_collapse;

/* ── Helpers ──────────────────────────────────────────────────────────── */

int visited_add(char ***visited, int *count, int *cap, const char *path) {
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

static void add_sidecar_entry(const char *glob, const char *summary, int collapse) {
    if (g_n_sidecar >= g_cap_sidecar) {
        g_cap_sidecar = g_cap_sidecar ? g_cap_sidecar * 2 : 16;
        g_sidecar = realloc(g_sidecar, (size_t)g_cap_sidecar * sizeof(*g_sidecar));
    }
    struct sidecar_entry *e = &g_sidecar[g_n_sidecar++];
    e->glob = strdup(glob);
    e->summary = summary ? strdup(summary) : NULL;
    e->collapse = collapse;
}

static void load_sidecar(const char *start_dir) {
    char dir[PATH_MAX];
    char path[PATH_MAX];

    strncpy(dir, start_dir, PATH_MAX - 1);
    dir[PATH_MAX - 1] = '\0';

    /* Remove trailing slash */
    size_t len = strlen(dir);
    while (len > 1 && dir[len - 1] == '/') dir[--len] = '\0';

    FILE *f = NULL;
    for (;;) {
        snprintf(path, sizeof(path), "%s/.archmap", dir);
        f = fopen(path, "r");
        if (f) break;
        char *slash = strrchr(dir, '/');
        if (!slash || slash == dir) break;
        *slash = '\0';
    }
    if (!f) return;

    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        size_t ll = strlen(line);
        if (ll > 0 && line[ll - 1] == '\n') line[--ll] = '\0';

        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0') continue;

        if (strncmp(p, "summary ", 8) == 0) {
            p += 8;
            while (*p == ' ' || *p == '\t') p++;
            char *glob_end = p;
            while (*glob_end && *glob_end != ' ' && *glob_end != '\t') glob_end++;
            if (!*glob_end) continue;
            *glob_end = '\0';
            char *text = glob_end + 1;
            while (*text == ' ' || *text == '\t') text++;
            if (*text) add_sidecar_entry(p, text, 0);
        } else if (strncmp(p, "collapse ", 9) == 0) {
            p += 9;
            while (*p == ' ' || *p == '\t') p++;
            char *end = p + strlen(p);
            while (end > p && (end[-1] == ' ' || end[-1] == '\t')) end--;
            *end = '\0';
            if (*p) add_sidecar_entry(p, NULL, 1);
        }
    }
    fclose(f);
}

static void write_file_list(const char *prefix_dir) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s.archmap.files", prefix_dir);
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < g_n_files; i++)
        fprintf(f, "%s\n", g_files[i].rel_path);
    fclose(f);
}

static const char *find_summary(const char *rel_path) {
    const char *result = NULL;
    for (int i = 0; i < g_n_sidecar; i++) {
        if (g_sidecar[i].summary &&
            fnmatch(g_sidecar[i].glob, rel_path, 0) == 0)
            result = g_sidecar[i].summary;
    }
    return result;
}

static int is_collapsed(const char *rel_path) {
    int collapsed = 0;
    for (int i = 0; i < g_n_sidecar; i++) {
        if (g_sidecar[i].collapse &&
            fnmatch(g_sidecar[i].glob, rel_path, 0) == 0)
            collapsed = 1;
    }
    for (int i = 0; i < opt_n_collapse; i++) {
        if (fnmatch(opt_collapse_globs[i], rel_path, 0) == 0)
            collapsed = 1;
    }
    for (int i = 0; i < opt_n_expand; i++) {
        if (fnmatch(opt_expand_globs[i], rel_path, 0) == 0)
            collapsed = 0;
    }
    return collapsed;
}

static int is_suppressed_fwd(struct symbol *s) {
    if (s->is_fwd_decl && s->tag_name) {
        if (saha_contains(&g_tags, s->tag_name, strlen(s->tag_name)))
            return 1;
    }
    return 0;
}

void print_tree(void) {
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

    write_file_list(prefix);
    load_sidecar(prefix);

    for (int i = 0; i < g_n_files; i++) {
        struct file_entry *fe = &g_files[i];

        const char *summary = find_summary(fe->rel_path);
        int collapsed = is_collapsed(fe->rel_path);

        /* Check if file has any visible symbols */
        int has_visible = 0;
        for (int j = 0; j < fe->n_syms; j++) {
            if (!is_suppressed_fwd(&fe->syms[j])) {
                has_visible = 1;
                break;
            }
        }
        if (!has_visible && !summary && !collapsed) continue;

        if (summary)
            printf("--- %s  @summary: %s\n", fe->rel_path, summary);
        else
            printf("--- %s\n", fe->rel_path);

        if (collapsed || !has_visible) continue;

        for (int sec = 0; sec < SEC_COUNT; sec++) {
            /* Check if this section has any visible symbols */
            int has_sec = 0;
            for (int j = 0; j < fe->n_syms; j++) {
                struct symbol *s = &fe->syms[j];
                if (s->section == sec && !is_suppressed_fwd(s)) {
                    has_sec = 1;
                    break;
                }
            }
            if (!has_sec) continue;

            printf("%s\n", section_headers[sec]);

            for (int j = 0; j < fe->n_syms; j++) {
                struct symbol *s = &fe->syms[j];
                if (s->section != sec) continue;
                if (is_suppressed_fwd(s)) continue;
                if (!s->text) continue;

                char *text = s->text;
                int text_len = (int)strlen(text);

                /* [def] section: strip leading #define for preproc captures */
                if (sec == SEC_DEF &&
                    (s->cap_idx == CAP_PREPROC_DEF || s->cap_idx == CAP_PREPROC_FUNC)) {
                    if (strncmp(text, "#define ", 8) == 0) {
                        text += 8;
                        text_len -= 8;
                    }
                }

                /* Strip struct/union keywords from all non-def sections */
                if (sec != SEC_DEF)
                    strip_struct_refs(text);

                /* Trim trailing whitespace/newline */
                while (text_len > 0 &&
                       (text[text_len-1] == '\n' || text[text_len-1] == '\r' ||
                        text[text_len-1] == ' ' || text[text_len-1] == '\t'))
                    text_len--;

                /* Print with line suffix */
                if (s->start_line == s->end_line)
                    printf("%.*s  //:%u\n", text_len, text, (unsigned)s->start_line);
                else
                    printf("%.*s  //:%u-%u\n", text_len, text,
                           (unsigned)s->start_line, (unsigned)s->end_line);

                /* Call edges */
                if (opt_show_calls && s->n_callees > 0) {
                    printf("  \xe2\x86\x92 ");
                    for (int k = 0; k < s->n_callees; k++) {
                        printf("%s%s", s->callees[k],
                               k < s->n_callees - 1 ? ", " : "\n");
                    }
                }
            }
        }
    }
}

void output_cleanup(void) {
    for (int i = 0; i < g_n_sidecar; i++) {
        free(g_sidecar[i].glob);
        free(g_sidecar[i].summary);
    }
    free(g_sidecar);
    g_sidecar = NULL;
    g_n_sidecar = g_cap_sidecar = 0;
}
