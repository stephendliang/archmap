#ifndef OUTPUT_H
#define OUTPUT_H

enum { SEC_STRUCT = 0, SEC_TYPE, SEC_DEF, SEC_DATA, SEC_FUNCTIONS, SEC_COUNT };

void print_tree(void);
void output_cleanup(void);
int visited_add(char ***visited, int *count, int *cap, const char *path);

/* Sidecar option arrays — populated by main.c's getopt loop */
extern char *opt_expand_globs[64];
extern int opt_n_expand;
extern char *opt_collapse_globs[64];
extern int opt_n_collapse;

#endif /* OUTPUT_H */
