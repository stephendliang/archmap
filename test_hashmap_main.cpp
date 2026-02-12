/*
 * Benchmark driver: SAHA tier-1 vs verstable vs boost::unordered_flat_set
 *
 * Usage: ./test_hashmap [N] [n_ops] [zipf_s]
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

#include <boost/unordered/unordered_flat_set.hpp>

extern "C" {

struct bench_result {
    double ins_mops;
    double pos_mops;
    double mix_mops;
    uint64_t unique;
    double dup_pct;
    double hit_pct;
};

void     bench_seed_rng(void);
uint64_t *bench_gen_zipf_keys(uint64_t n, double s, uint64_t n_ops);

struct bench_result bench_saha(const uint64_t *k_ins, const uint64_t *k_pos,
                               const uint64_t *k_mix, uint64_t n_ops);
struct bench_result bench_vt(const uint64_t *k_ins, const uint64_t *k_pos,
                             const uint64_t *k_mix, uint64_t n_ops);
struct bench_result bench_avx64(const uint64_t *k_ins, const uint64_t *k_pos,
                                const uint64_t *k_mix, uint64_t n_ops);

} // extern "C"

/* ================================================================
 * Timing (same as C side)
 * ================================================================ */

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * boost::unordered_flat_set benchmark
 * ================================================================ */

static bench_result bench_boost(const uint64_t *k_ins, const uint64_t *k_pos,
                                const uint64_t *k_mix, uint64_t n_ops) {
    bench_result r{};
    boost::unordered_flat_set<uint64_t> set;

    double t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++)
        set.insert(k_ins[i]);
    double elapsed = now_sec() - t0;
    r.ins_mops = (double)n_ops / elapsed / 1e6;
    r.unique   = set.size();

    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++)
        set.find(k_pos[i]);
    elapsed = now_sec() - t0;
    r.pos_mops = (double)n_ops / elapsed / 1e6;

    uint64_t hits = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_ops; i++) {
        if (set.find(k_mix[i]) != set.end())
            hits++;
    }
    elapsed = now_sec() - t0;
    r.mix_mops = (double)n_ops / elapsed / 1e6;
    r.hit_pct  = 100.0 * (double)hits / (double)n_ops;

    return r;
}

/* ================================================================
 * Main
 * ================================================================ */

static void print_row(const char *label, bench_result *r) {
    printf("  %-10s %6.1f Mops/s   %6.1f Mops/s   %6.1f Mops/s\n",
           label, r->ins_mops, r->pos_mops, r->mix_mops);
}

int main(int argc, char **argv) {
    uint64_t N     = 1000000;
    uint64_t n_ops = 10000000;
    double   zipf_s = 1.0;

    if (argc > 1) N     = (uint64_t)atol(argv[1]);
    if (argc > 2) n_ops = (uint64_t)atol(argv[2]);
    if (argc > 3) zipf_s = atof(argv[3]);

    printf("bench: N=%lu ops=%lu zipf_s=%.2f\n\n",
           (unsigned long)N, (unsigned long)n_ops, zipf_s);

    bench_seed_rng();

    uint64_t *k_ins = bench_gen_zipf_keys(N, zipf_s, n_ops);
    uint64_t *k_pos = bench_gen_zipf_keys(N, zipf_s, n_ops);
    uint64_t *k_mix = bench_gen_zipf_keys(2 * N, zipf_s, n_ops);

    bench_result r_saha  = bench_saha(k_ins, k_pos, k_mix, n_ops);
    bench_result r_avx64 = bench_avx64(k_ins, k_pos, k_mix, n_ops);
    bench_result r_vt    = bench_vt(k_ins, k_pos, k_mix, n_ops);
    bench_result r_boost = bench_boost(k_ins, k_pos, k_mix, n_ops);

    printf("              insert          lookup+         lookup±\n");
    print_row("saha",      &r_saha);
    print_row("avx_map64", &r_avx64);
    print_row("verstable", &r_vt);
    print_row("boost",     &r_boost);
    printf("\n");
    printf("  insert:  %lu unique of %lu (%.1f%% dup)\n",
           (unsigned long)r_saha.unique, (unsigned long)n_ops, r_saha.dup_pct);
    printf("  lookup±: %.1f%% hit\n", r_saha.hit_pct);

    free(k_ins);
    free(k_pos);
    free(k_mix);
    return 0;
}
