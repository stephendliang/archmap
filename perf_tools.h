#ifndef PERF_TOOLS_H
#define PERF_TOOLS_H

struct sym_resolver;
struct perf_opts;
struct perf_profile;

int run_uprof(struct perf_opts *opts, struct perf_profile *prof);
int run_mca(struct sym_resolver *sr, struct perf_opts *opts,
            struct perf_profile *prof);
int run_pahole(struct perf_opts *opts, struct perf_profile *prof,
               int has_debug);
void xref_skeleton(struct perf_opts *opts, struct perf_profile *prof);
int run_remarks(struct perf_opts *opts, struct perf_profile *prof);

#endif /* PERF_TOOLS_H */
