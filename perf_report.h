#ifndef PERF_REPORT_H
#define PERF_REPORT_H

struct perf_opts;
struct perf_profile;
struct run_stats;

/* Statistics */
void compute_stats(struct run_stats *rs);
double welch_t_test(struct run_stats *a, struct run_stats *b);

/* Output */
void print_report(struct perf_opts *opts, struct perf_profile *prof);
void print_comparison(struct perf_opts *opts,
                      struct perf_profile *prof_a, const char *a_cmd,
                      struct perf_profile *prof_b, const char *b_cmd);

#endif /* PERF_REPORT_H */
