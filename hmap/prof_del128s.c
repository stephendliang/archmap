/*
 * prof_del128s: Isolated delete-all workload for AMDuProf profiling.
 * Compile with -DUSE_OLD for rehash version, without for displacement version.
 * Runs 5 rounds of insert-all + delete-all to give the profiler enough samples.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef USE_OLD
#include "avx_map128s_old.h"
#else
#include "avx_map128s.h"
#endif

#define N 2000000
#define ROUNDS 5

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main(void) {
    uint64_t s1 = 0xdeadbeefcafe1234ULL;
    uint64_t s2 = 0x0123456789abcdefULL;
    uint64_t *klo = malloc(N * sizeof(uint64_t));
    uint64_t *khi = malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        klo[i] = splitmix64(&s1);
        khi[i] = splitmix64(&s2);
    }

    volatile int sink = 0;
    for (int r = 0; r < ROUNDS; r++) {
        struct avx_map128s m;
        avx_map128s_init(&m);
        for (int i = 0; i < N; i++)
            avx_map128s_insert(&m, klo[i], khi[i]);
        for (int i = 0; i < N; i++)
            sink += avx_map128s_delete(&m, klo[i], khi[i]);
        avx_map128s_destroy(&m);
    }

    free(klo); free(khi);
    printf("sink=%d\n", sink);
    return 0;
}
