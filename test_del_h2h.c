/*
 * Head-to-head: baseline vs tombstone with direct function calls (no vtable).
 * This eliminates function pointer overhead to get true performance comparison.
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512bw -std=gnu11 \
 *       -o test_del_h2h test_del_h2h.c -lm
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>

#include "avx_map64.h"

/* ================================================================
 * Tombstone variant (inlined, same compilation unit)
 * ================================================================ */

#define TOMBSTONE 1ULL

struct avx_tomb {
    uint64_t *keys;
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
    uint32_t n_tombstones;
};

static inline uint32_t tomb_hash(uint64_t key) {
    return (uint32_t)_mm_crc32_u64(0, key);
}

static size_t tomb_mapsize(uint32_t cap) {
    size_t raw = (size_t)cap * sizeof(uint64_t);
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void tomb_alloc(struct avx_tomb *m, uint32_t cap) {
    size_t total = tomb_mapsize(cap);
    m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (m->keys == MAP_FAILED)
        m->keys = (uint64_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
    m->n_tombstones = 0;
}

static inline void tomb_init(struct avx_tomb *m) { memset(m, 0, sizeof(*m)); }

static inline void tomb_destroy(struct avx_tomb *m) {
    if (m->keys) munmap(m->keys, tomb_mapsize(m->cap));
}

static inline void tomb_prefetch(struct avx_tomb *m, uint64_t key) {
    uint32_t gi = tomb_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static inline void tomb_prefetch2(struct avx_tomb *m, uint64_t key) {
    uint32_t gi = tomb_hash(key) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
    gi = (gi + 1) & m->mask;
    _mm_prefetch((const char *)(m->keys + (gi << 3)), _MM_HINT_T0);
}

static void tomb_rehash(struct avx_tomb *m, uint32_t new_cap) {
    uint32_t old_cap = m->cap;
    uint64_t *old_keys = m->keys;
    tomb_alloc(m, new_cap);
    uint32_t mask = m->mask;
    for (uint32_t i = 0; i < old_cap; i++) {
        uint64_t key = old_keys[i];
        if (!key || key == TOMBSTONE) continue;
        uint32_t gi = tomb_hash(key) & mask;
        for (;;) {
            uint64_t *grp = m->keys + (gi << 3);
            __mmask8 em = _mm512_testn_epi64_mask(
                _mm512_load_si512((const __m512i *)grp),
                _mm512_load_si512((const __m512i *)grp));
            if (em) { grp[__builtin_ctz(em)] = key; m->count++; break; }
            gi = (gi + 1) & mask;
        }
    }
    munmap(old_keys, tomb_mapsize(old_cap));
}

static inline int tomb_contains(struct avx_tomb *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = tomb_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle))   return 1;
        if (_mm512_testn_epi64_mask(group, group))     return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int tomb_insert(struct avx_tomb *m, uint64_t key) {
    if (m->cap == 0) tomb_alloc(m, 64);
    if (m->count * 4 >= m->cap * 3)
        tomb_rehash(m, m->cap * 2);
    else if ((m->count + m->n_tombstones) * 8 >= m->cap * 7)
        tomb_rehash(m, m->cap);

    uint32_t gi = tomb_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    __m512i tomb_v = _mm512_set1_epi64((long long)TOMBSTONE);
    int ins_gi = -1, ins_slot = -1;

    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((const __m512i *)grp);
        if (_mm512_cmpeq_epi64_mask(group, needle)) return 0;
        if (ins_gi < 0) {
            __mmask8 avail = _mm512_testn_epi64_mask(group, group)
                           | _mm512_cmpeq_epi64_mask(group, tomb_v);
            if (avail) { ins_gi = (int)gi; ins_slot = __builtin_ctz(avail); }
        }
        if (_mm512_testn_epi64_mask(group, group)) break;
        gi = (gi + 1) & m->mask;
    }

    uint64_t *grp = m->keys + (ins_gi << 3);
    if (grp[ins_slot] == TOMBSTONE) m->n_tombstones--;
    grp[ins_slot] = key;
    m->count++;
    return 1;
}

static inline int tomb_delete(struct avx_tomb *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = tomb_hash(key) & m->mask;
    __m512i needle = _mm512_set1_epi64((long long)key);
    for (;;) {
        uint64_t *grp = m->keys + (gi << 3);
        __m512i group = _mm512_load_si512((__m512i *)grp);
        __mmask8 mm = _mm512_cmpeq_epi64_mask(group, needle);
        if (mm) {
            grp[__builtin_ctz(mm)] = TOMBSTONE;
            m->count--;
            m->n_tombstones++;
            return 1;
        }
        if (_mm512_testn_epi64_mask(group, group)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* ================================================================
 * RNG / Zipf / Timing (identical to test_hashmap.c)
 * ================================================================ */

static uint64_t rng_s[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t xoshiro256ss(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return result;
}
static void seed_rng(void) {
    uint64_t s = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double *zipf_cdf;
static uint64_t zipf_n;
static void zipf_setup(uint64_t n, double s) {
    zipf_n = n;
    zipf_cdf = (double *)malloc(n * sizeof(double));
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += 1.0 / pow((double)(i + 1), s);
        zipf_cdf[i] = sum;
    }
    for (uint64_t i = 0; i < n; i++) zipf_cdf[i] /= sum;
}
static inline uint64_t zipf_sample(void) {
    double u = (xoshiro256ss() >> 11) * 0x1.0p-53;
    uint64_t lo = 0, hi = zipf_n - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (zipf_cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    return lo + 1;
}

#define PF_DIST 12

/* ================================================================
 * Per-variant benchmark with DIRECT calls (no function pointers)
 * ================================================================ */

struct bench_result {
    double del_mops;
    double mixed_mops;
    int verified;
};

static struct bench_result bench_baseline(uint64_t pool_size, uint64_t n_mixed_ops,
                                          double zipf_s,
                                          int pct_lookup, int pct_insert, int pct_delete) {
    struct bench_result r = {0, 0, 1};
    int thresh_ins = pct_lookup + pct_insert;
    (void)pct_delete;

    struct avx_map64 m;
    avx_map64_init(&m);

    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 2;
        if (avx_map64_insert(&m, k)) pool[gen++] = k;
    }
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }

    /* pure delete */
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < pool_size; i++) {
        if (i + PF_DIST < pool_size)
            avx_map64_prefetch2(&m, pool[i + PF_DIST]);
        tot += (uint64_t)avx_map64_delete(&m, pool[i]);
    }
    r.del_mops = (double)pool_size / (now_sec() - t0) / 1e6;
    if (m.count != 0 || tot != pool_size) r.verified = 0;
    avx_map64_destroy(&m);

    /* mixed workload */
    avx_map64_init(&m);
    uint32_t live = (uint32_t)(pool_size / 2), total = (uint32_t)pool_size;
    for (uint32_t i = 0; i < live; i++) avx_map64_insert(&m, pool[i]);
    zipf_setup(live, zipf_s);

    uint64_t *op_keys = (uint64_t *)malloc(n_mixed_ops * sizeof(uint64_t));
    uint8_t  *op_type = (uint8_t *)malloc(n_mixed_ops);

    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
        if (pct < (uint32_t)pct_lookup && live > 0) {
            op_type[i] = 0; op_keys[i] = pool[(zipf_sample() - 1) % live];
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
            op_keys[i] = pool[di];
            uint64_t tmp = pool[di]; pool[di] = pool[live]; pool[live] = tmp; live++;
        } else if (live > 2) {
            op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
            op_keys[i] = pool[li]; live--;
            uint64_t tmp = pool[li]; pool[li] = pool[live]; pool[live] = tmp;
        } else {
            op_type[i] = 0;
            op_keys[i] = (live > 0) ? pool[(uint32_t)(xoshiro256ss() >> 32) % live]
                                     : (xoshiro256ss() | 2);
        }
    }
    free(zipf_cdf); zipf_cdf = NULL;

    tot = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        if (i + PF_DIST < n_mixed_ops) {
            if (op_type[i + PF_DIST] == 0)
                avx_map64_prefetch(&m, op_keys[i + PF_DIST]);
            else
                avx_map64_prefetch2(&m, op_keys[i + PF_DIST]);
        }
        switch (op_type[i]) {
            case 0: tot += (uint64_t)avx_map64_contains(&m, op_keys[i]); break;
            case 1: tot += (uint64_t)avx_map64_insert(&m, op_keys[i]); break;
            case 2: tot += (uint64_t)avx_map64_delete(&m, op_keys[i]); break;
        }
    }
    r.mixed_mops = (double)n_mixed_ops / (now_sec() - t0) / 1e6;

    if (m.count != live) r.verified = 0;
    if (r.verified) {
        for (uint32_t i = 0; i < live; i++)
            if (!avx_map64_contains(&m, pool[i])) { r.verified = 0; break; }
    }
    if (r.verified && total > live) {
        for (uint32_t i = live; i < total; i++)
            if (avx_map64_contains(&m, pool[i])) { r.verified = 0; break; }
    }

    (void)tot;
    free(op_keys); free(op_type);
    avx_map64_destroy(&m);
    free(pool);
    return r;
}

static struct bench_result bench_tombstone(uint64_t pool_size, uint64_t n_mixed_ops,
                                            double zipf_s,
                                            int pct_lookup, int pct_insert, int pct_delete) {
    struct bench_result r = {0, 0, 1};
    int thresh_ins = pct_lookup + pct_insert;
    (void)pct_delete;

    struct avx_tomb m;
    tomb_init(&m);

    uint64_t *pool = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
    uint64_t gen = 0;
    while (gen < pool_size) {
        uint64_t k = xoshiro256ss() | 2;
        if (tomb_insert(&m, k)) pool[gen++] = k;
    }
    for (uint64_t i = pool_size - 1; i > 0; i--) {
        uint64_t j = xoshiro256ss() % (i + 1);
        uint64_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }

    /* pure delete */
    uint64_t tot = 0;
    double t0 = now_sec();
    for (uint64_t i = 0; i < pool_size; i++) {
        if (i + PF_DIST < pool_size)
            tomb_prefetch2(&m, pool[i + PF_DIST]);
        tot += (uint64_t)tomb_delete(&m, pool[i]);
    }
    r.del_mops = (double)pool_size / (now_sec() - t0) / 1e6;
    if (m.count != 0 || tot != pool_size) r.verified = 0;
    tomb_destroy(&m);

    /* mixed workload */
    tomb_init(&m);
    uint32_t live = (uint32_t)(pool_size / 2), total = (uint32_t)pool_size;
    for (uint32_t i = 0; i < live; i++) tomb_insert(&m, pool[i]);
    zipf_setup(live, zipf_s);

    uint64_t *op_keys = (uint64_t *)malloc(n_mixed_ops * sizeof(uint64_t));
    uint8_t  *op_type = (uint8_t *)malloc(n_mixed_ops);

    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        uint32_t pct = (uint32_t)(xoshiro256ss() >> 32) % 100;
        if (pct < (uint32_t)pct_lookup && live > 0) {
            op_type[i] = 0; op_keys[i] = pool[(zipf_sample() - 1) % live];
        } else if (pct < (uint32_t)thresh_ins && live < total) {
            op_type[i] = 1;
            uint32_t di = live + (uint32_t)(xoshiro256ss() >> 32) % (total - live);
            op_keys[i] = pool[di];
            uint64_t tmp = pool[di]; pool[di] = pool[live]; pool[live] = tmp; live++;
        } else if (live > 2) {
            op_type[i] = 2;
            uint32_t li = (uint32_t)(xoshiro256ss() >> 32) % live;
            op_keys[i] = pool[li]; live--;
            uint64_t tmp = pool[li]; pool[li] = pool[live]; pool[live] = tmp;
        } else {
            op_type[i] = 0;
            op_keys[i] = (live > 0) ? pool[(uint32_t)(xoshiro256ss() >> 32) % live]
                                     : (xoshiro256ss() | 2);
        }
    }
    free(zipf_cdf); zipf_cdf = NULL;

    tot = 0;
    t0 = now_sec();
    for (uint64_t i = 0; i < n_mixed_ops; i++) {
        if (i + PF_DIST < n_mixed_ops) {
            if (op_type[i + PF_DIST] == 0)
                tomb_prefetch(&m, op_keys[i + PF_DIST]);
            else
                tomb_prefetch2(&m, op_keys[i + PF_DIST]);
        }
        switch (op_type[i]) {
            case 0: tot += (uint64_t)tomb_contains(&m, op_keys[i]); break;
            case 1: tot += (uint64_t)tomb_insert(&m, op_keys[i]); break;
            case 2: tot += (uint64_t)tomb_delete(&m, op_keys[i]); break;
        }
    }
    r.mixed_mops = (double)n_mixed_ops / (now_sec() - t0) / 1e6;

    if (m.count != live) {
        fprintf(stderr, "tomb FAIL: count=%u expected=%u (tombs=%u)\n",
                m.count, live, m.n_tombstones);
        r.verified = 0;
    }
    if (r.verified) {
        for (uint32_t i = 0; i < live; i++)
            if (!tomb_contains(&m, pool[i])) { r.verified = 0; break; }
    }
    if (r.verified && total > live) {
        for (uint32_t i = live; i < total; i++)
            if (tomb_contains(&m, pool[i])) { r.verified = 0; break; }
    }

    (void)tot;
    free(op_keys); free(op_type);
    tomb_destroy(&m);
    free(pool);
    return r;
}

/* ================================================================
 * Main: run 3 iterations of each profile, report median
 * ================================================================ */

int main(void) {
    uint64_t pool_size   = 1000000;
    uint64_t n_mixed_ops = 5000000;
    double   zipf_s      = 0.8;

    struct { const char *name; int lkp, ins, del; } profiles[] = {
        { "read-heavy",  90,  5,  5 },
        { "balanced",    50, 25, 25 },
        { "churn",       33, 33, 34 },
        { "write-heavy", 10, 50, 40 },
        { "eviction",    20, 10, 70 },
    };
    int n_profiles = sizeof(profiles) / sizeof(profiles[0]);
    int n_iter = 3;

    printf("Direct-call head-to-head (pool=%luK, mixed=%luM, %d iterations)\n",
           pool_size/1000, n_mixed_ops/1000000, n_iter);
    printf("=========================================================================\n\n");

    printf("%-12s  %20s  %20s  %8s\n", "", "baseline", "tombstone", "speedup");
    printf("%-12s  %9s %9s  %9s %9s  %8s\n",
           "profile", "pure-del", "mixed", "pure-del", "mixed", "mixed");
    for (int i = 0; i < 78; i++) putchar('-');
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        double base_del[3], base_mix[3], tomb_del[3], tomb_mix[3];
        int base_ok = 1, tomb_ok = 1;

        for (int it = 0; it < n_iter; it++) {
            seed_rng();
            struct bench_result rb = bench_baseline(pool_size, n_mixed_ops, zipf_s,
                profiles[p].lkp, profiles[p].ins, profiles[p].del);
            base_del[it] = rb.del_mops;
            base_mix[it] = rb.mixed_mops;
            if (!rb.verified) base_ok = 0;

            seed_rng();
            struct bench_result rt = bench_tombstone(pool_size, n_mixed_ops, zipf_s,
                profiles[p].lkp, profiles[p].ins, profiles[p].del);
            tomb_del[it] = rt.del_mops;
            tomb_mix[it] = rt.mixed_mops;
            if (!rt.verified) tomb_ok = 0;
        }

        /* sort for median */
        for (int i = 0; i < 2; i++) for (int j = i+1; j < 3; j++) {
            if (base_del[i] > base_del[j]) { double t = base_del[i]; base_del[i] = base_del[j]; base_del[j] = t; }
            if (base_mix[i] > base_mix[j]) { double t = base_mix[i]; base_mix[i] = base_mix[j]; base_mix[j] = t; }
            if (tomb_del[i] > tomb_del[j]) { double t = tomb_del[i]; tomb_del[i] = tomb_del[j]; tomb_del[j] = t; }
            if (tomb_mix[i] > tomb_mix[j]) { double t = tomb_mix[i]; tomb_mix[i] = tomb_mix[j]; tomb_mix[j] = t; }
        }

        double bd = base_del[1], bm = base_mix[1], td = tomb_del[1], tm = tomb_mix[1];
        printf("%-12s  %7.1f%s %7.1f%s  %7.1f%s %7.1f%s  %+6.1f%%\n",
               profiles[p].name,
               bd, base_ok ? "  " : "! ", bm, base_ok ? "  " : "! ",
               td, tomb_ok ? "  " : "! ", tm, tomb_ok ? "  " : "! ",
               100.0 * (tm - bm) / bm);
    }

    printf("\n");
    return 0;
}
