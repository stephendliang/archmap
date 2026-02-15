# Prefetch Distance Optimization for AVX-512 Open-Addressing Hash Maps

## Abstract

We investigate prefetch distance (PF_DIST) tuning for `avx_map64`, an AVX-512
SIMD open-addressing hash map with backshift deletion.  Through systematic
exploration of four delete strategies, instrumented backshift analysis, a
PF_DIST sweep across five workload profiles, three adaptive PF approaches
(clock_gettime, RDTSC, counter-based), and a hyperparameter sweep with
counter decay variants, we find:

1. **Prefetch distance is the dominant optimization lever** for mixed
   workloads — not the deletion algorithm itself.
2. **Adaptation overhead exceeds its benefit** for constant-ratio workloads.
   Controlled A/B testing showed all adaptive variants (hard reset, >>1 decay,
   >>2 decay, various windows/thresholds) perform 10-20% worse than fixed PF.
3. **PF=4 is universally optimal** for mixed workloads with inlined hashmap
   ops; PF=12 is optimal for pure-delete and lookup-only phases.
4. **The Sieve cache replacement principle** (simple counters beat precise
   measurement) has a corollary: **when the signal has zero entropy
   (constant-ratio workloads), even a 1-bit counter is wasted work.**

## 1. Experimental Setup

**Hardware**: AMD Zen 5 (Family 0x1a, Model 0x11), 2 cores, KVM guest.
Linux 6.18.7-arch1-1.  All benchmarks pinned to core 0 via `taskset -c 0`.

**Map**: `avx_map64` — 8-wide groups in 64-byte cache lines, CRC32C hash
(hardware `crc32q`), 75% max load factor, linear probing with backshift
deletion.  Keys are `uint64_t`; zero is the empty sentinel.

**Workload generation**: Zipf(s=1.0) key distribution over a pool of N unique
keys.  Pool is partitioned into live (in map) and dead (not in map) segments.
Operations are pre-generated with accurate pool tracking for post-hoc
verification.

**Profiles tested**:

| Profile | Lookup/Insert/Delete | Real-world analog |
|---|---|---|
| read-heavy | 90/5/5 | Session cache, DNS cache |
| balanced | 50/25/25 | General-purpose KV store |
| churn | 33/33/34 | Connection tracker |
| write-heavy | 10/50/40 | Bulk ingestion + TTL eviction |
| eviction | 20/10/70 | Cache eviction storm |

**Verification**: Every profile run verifies final map state against the pool
partition (all live keys present, no dead keys present, count matches).

## 2. Delete Strategy Exploration

### 2.1 Variants Tested

Four deletion strategies were implemented and benchmarked (`test_del_explore.c`):

**A. Backshift (baseline)**: Find key, zero it, scan forward for displaced keys
that can be moved back.  Uses pipelined CRC32 — all occupied keys in the scan
group are hashed independently before checking candidates, breaking the serial
dependency chain (CRC32: 3-cycle latency, 1-cycle throughput on Zen).

**B. Pure tombstone**: Mark deleted keys with sentinel value (key=1).  No
backshift.  Periodic compaction (full rehash) when effective load
(count + tombstones) exceeds 87.5% of capacity.

**C. Hybrid**: 1-group bounded backshift.  If no movable candidate found in the
immediate next group, insert a tombstone instead.  Bounds worst-case backshift
cost while maintaining short probe chains for most keys.

**D. Displacement bitmap**: 1 byte per group tracking which slots hold keys
displaced from their home group.  Allows backshift to skip at-home keys without
hashing.

### 2.2 Instrumentation Findings

Instrumenting the baseline backshift on 1M keys revealed:

- **98.4%** of deletes require NO backshift (hole is at end of probe chain)
- Only **1.6%** trigger backshift, with average chain length **1.02** groups
- **86.5%** of CRC32 hashes during backshift are wasted on at-home keys
- Maximum observed backshift chain: **5 groups**

**Implication**: Backshift is already near-free at 75% load factor.  Optimizing
the deletion algorithm itself yields diminishing returns — the bottleneck is
elsewhere.

### 2.3 Vtable vs. Direct-Call Artifact

Initial benchmarks used function-pointer dispatch (vtable pattern) for fair
variant comparison.  Tombstone showed +10–29% gains over baseline.

A direct-call head-to-head (`test_del_h2h.c`) with fully inlined functions
revealed the vtable gains were artifacts: **function-pointer overhead penalizes
the baseline's larger code path disproportionately** compared to tombstone's
smaller code path.  With inlining, tombstone was within ±5% of baseline, with
high variance (read-heavy: −3% to −23% across runs).

**Lesson**: Micro-benchmark dispatch mechanism can dominate the signal.  Always
validate with inlined/direct calls before drawing conclusions.

### 2.4 Displacement Bitmap

Maintaining a per-group byte tracking displaced keys adds ~2 instructions to
every insert (read-modify-write the bitmap byte).  Since only 1.6% of deletes
benefit from skipping at-home key hashes during backshift, the insert overhead
far exceeds the delete savings.  Net effect: **−3% to −5% regression** on all
profiles.

## 3. PF_DIST as the Dominant Parameter

### 3.1 Sweep Results

A systematic sweep (`test_pfdist.c`) over PF_DIST ∈ {4, 6, 8, 10, 12, 14, 16,
20, 24, 32} revealed that different workload profiles have different optimal
prefetch distances.

Official benchmark results (separate compilation, no cross-TU inlining):

| Profile | PF=4 | PF=8 | PF=12 | Best |
|---|---|---|---|---|
| pure-delete | 69.1 | 77.2 | **81.2** | PF=12 |
| read-heavy | 161.7 | 167.4 | **171.6** | PF=12 |
| balanced | 57.2 | 54.1 | **57.9** | PF=12 |
| churn | **56.5** | 51.7 | 52.1 | PF=4 |
| write-heavy | **42.4** | 40.9 | 39.8 | PF=4 |
| eviction | **129.0** | 120.6 | 116.6 | PF=4 |

### 3.2 Physical Interpretation

**Read-heavy / high-load**: Probe chains average ~1.5 groups.  Each lookup
touches 1–2 cache lines.  At ~20 ns/op, PF_DIST=12 gives 240 ns lead time
— enough for an L3 hit (~30–40 ns) plus memory-level parallelism.  The CPU
has 12 ops of independent work to overlap with the prefetch.

**Mutation-heavy / low-load**: Inserts and deletes have more complex control
flow (SIMD compare, conditional store, possible resize/backshift).  Per-op
latency is higher (~25–40 ns), so PF_DIST=12 prefetches too far ahead —
the prefetched line may be evicted before use.  PF_DIST=4 at ~100–160 ns
lead time matches the actual L2/L3 latency better.

**Crossover**: At ~50% mutation fraction, PF=12 and PF=4 are approximately
equal (balanced profile: 57.9 vs 57.2, within noise).

## 4. Adaptive PF_DIST Design

### 4.1 Failed Approach: Throughput-Based (clock_gettime)

The initial adaptive implementation measured throughput every 1024 ops using
`clock_gettime(CLOCK_MONOTONIC)` and computed:

```
pf_dist = target_lead_ns / ns_per_op
```

This regressed on all profiles (−5% to −30%):

1. **Self-referential feedback loop**: PF_DIST affects throughput, which we
   use to compute PF_DIST.  The system oscillates rather than converging.
   Increasing PF_DIST → more prefetch misses → lower throughput → lower
   computed PF_DIST → fewer prefetch misses → higher throughput → higher
   computed PF_DIST → cycle repeats.

2. **Measurement overhead**: Even with VDSO, `clock_gettime` costs ~15–25 ns
   (VDSO fast path involves reading the kernel's vsyscall page).  At 1024-op
   windows with ~20 ns/op, that's ~1.2% overhead per measurement — and the
   jitter from the measurement itself corrupts the signal.

3. **Serialization**: The timing call acts as a pipeline fence, disrupting
   out-of-order execution exactly when we need maximum MLP (memory-level
   parallelism).

### 4.2 Rejected Alternative: RDTSC

`rdtsc` reads the Time Stamp Counter in ~20–25 cycles without a syscall.
However:

- On AMD Zen, `rdtsc` is not serializing.  Accurate measurement requires
  `lfence; rdtsc` (or `rdtscp`), which **forces retirement of all prior
  instructions** — exactly the OoO disruption we're trying to avoid.
- Even without `lfence`, the `rdtsc` result arrives ~25 cycles late due to
  pipeline depth, making per-window measurements noisy.
- The fundamental feedback-loop problem (§4.1 point 1) remains regardless
  of timing source.

### 4.3 Counter-Based Approach (Sieve Analogy)

The key insight: **we don't need to measure how fast we're going.  We need
to know what kind of work we're doing.**  This is a qualitative signal
(mutation-heavy vs. read-heavy), not a quantitative one (ns/op).

**Design**:
- Pure delete: Check load factor every 1024 ops.  `m.count * 2 > m.cap`
  (>50% load) → PF=12, else PF=4.
- Mixed workload: Count mutations in a 1024-op window.
  `mut_count > 512` (>50% mutations) → PF=4, else PF=12.

**Cost**: One `add` instruction per iteration (`mut_count += (op_type[i] != 0)`).
Compiles to `cmp` + `adc` or `cmov` — branchless, 1 cycle, no serialization.

**Sieve/CLOCK analogy**: This mirrors cache replacement algorithm evolution.
LHD (Least Hit Density) and ARC track per-object hit rates with precise
timestamps.  Sieve uses a single bit per entry: dramatically simpler, yet
matches or beats LHD.  The reason: **the marginal information gain from
precise measurement doesn't justify the overhead in the critical path.**

Initial cross-session benchmarks appeared to show +3% to +19% gains for the
counter-based adaptive approach vs. fixed PF=12.  However, this comparison
was unreliable (see §4.4).

### 4.4 Hyperparameter Sweep and Counter Decay

A systematic sweep (`test_pf_sweep.c`) explored the parameter space:

- **Counter decay**: hard reset (0), >>1, >>2
- **Window size**: 256, 512, 1024, 2048
- **Mutation threshold**: 30%, 40%, 50%, 60%, 70%
- **PF pair**: {3,12}, {4,10}, {4,12}, {4,14}, {6,12}, {6,14}
- **Load factor threshold** (pure delete): >25%, >33%, >50%, >67%

**Key findings**:

1. **Flat landscape**: All decay mechanisms, window sizes, and thresholds
   produce results within ±3% of each other.  Hard reset ≈ >>1 decay ≈
   >>2 decay.  There are no significantly better hyperparameters.

2. **Decay math**: With >>s shift decay, the counter at steady state
   converges to `f × WIN` (where f = mutation fraction), regardless of s.
   This means the threshold values are equivalent across decay modes after
   normalization.  Decay only adds value for workloads that change character
   over time — which our constant-ratio benchmarks do not.

### 4.5 Controlled A/B: Adaptation is a Net Negative

The cross-session comparisons that showed adaptive winning were unreliable.
A controlled A/B test (`test_pf_ab2.c`) with order-rotated execution
(each variant runs first in one of 4 interleaved rounds, eliminating cache
ordering bias) and `__attribute__((noinline))` to prevent cross-function
optimization revealed:

**Mixed workloads** (4-run average, order-rotated, noinline):

| Profile | fixed12 | **fixed4** | adapt(reset) | adapt(>>1) |
|---|---|---|---|---|
| read-heavy | 114.9 | **122.9** | 102.3 | 99.7 |
| balanced | 56.7 | **60.3** | 49.6 | 47.5 |
| churn | 56.0 | **58.5** | 51.6 | 51.3 |
| write-heavy | 48.9 | **50.0** | 44.5 | 44.8 |
| eviction | 132.1 | **163.4** | 118.1 | 118.0 |

**Pure delete** (4-run average):
```
fixed12 = 90.3    LF>50% = 78.3    LF>25% = 87.4
```

**Analysis**:

1. **Fixed PF=4 wins every mixed profile** — even read-heavy (+7% over
   PF=12).  With inlined hashmap ops, per-iteration latency is ~7-10ns,
   so PF=4's ~28-40ns lead time suffices for L2 hits (~10ns) and most L3
   hits (~30-40ns).  PF=12's ~84-120ns lead time is excessive — prefetched
   lines may be evicted from L1d before the op reaches them.

2. **Both adaptive variants lose by 10-20%** on every profile.  The
   adaptation overhead — counter increment, variable `pf_dist` (vs.
   compile-time constant), boundary check — disrupts the tight inner loop.
   The variable pf_dist forces the compiler to compute `i + pf_dist`
   dynamically (vs. `i + 4` which folds into addressing), and prevents
   certain loop optimizations.

3. **The Sieve principle has a corollary**: if the signal has zero entropy
   (constant-ratio workloads), even a 1-bit counter is wasted work.
   The overhead of asking "what kind of work am I doing?" exceeds the
   value of the answer when the answer never changes.

### 4.6 Final Design

Based on the controlled A/B evidence:
- **Pure delete**: Fixed PF=12 (probe chains are long at high load,
  need maximum look-ahead)
- **Mixed workloads**: Fixed PF=4 (shorter lead time matches inlined
  per-op latency; avoids L1d eviction from over-prefetching)
- **Insert/lookup-only**: Fixed PF=12 (simple loop with low per-op
  latency benefits from longer look-ahead)

No runtime adaptation.  The optimal PF is determined by the operation
mix, which is known at the call site.  Users running dynamic workloads
can call `avx_map64_prefetch` or `avx_map64_prefetch2` explicitly with
application-appropriate distances.

## 5. AMD uProf Profiling

### 5.1 Data Access Profile

`bench_avx64_del` (delete + mixed workload benchmark):

| Metric | Value |
|---|---|
| CPI | 1.33 |
| L1 DC accesses (PTI) | 669 |
| L1 DC miss rate | 8.74% |
| L1 DC refills from L2 (PTI) | 31.8 |
| L1 DC refills from L3 (PTI) | 28.1 |
| L1 DC refills from DRAM (PTI) | 0.35 |
| L1 DTLB misses (PTI) | 17.7 |
| L2 DTLB misses (PTI) | 0.53 |

**Prefetch effectiveness**: The 8.74% L1 DC miss rate (vs. what would be
~30–40% without prefetching at this table size) confirms the adaptive
prefetch is working.  Nearly all refills come from L2/L3 — no DRAM latency.

**TLB pressure**: 17.7 L1 DTLB misses per 1000 instructions is significant.
At 1M keys, the table spans ~16 MB across ~4096 4KB pages.  L1 DTLB on Zen 5
has 64 entries — random access patterns guarantee frequent misses.  Huge pages
(2MB) would reduce the table to ~8 TLB entries, eliminating this bottleneck.

### 5.2 Performance Assessment (Extended)

| Metric | bench_avx64_del | bench_avx64 (insert/lookup) |
|---|---|---|
| CPI | 1.31 | 0.74 |
| Branch mispredict rate | **11.16%** | 1.85% |
| Ineffective SW PF (PTI) | 5.6 | 50.7 |
| STLI (PTI) | 2.0 | 0.0 |

**Branch misprediction**: The 11.16% misprediction rate in the mixed workload
is 6× higher than the insert/lookup-only benchmark.  This is caused by the
3-way `switch(op_type[i])` dispatch — with balanced profiles (33/33/34), the
branch predictor has no pattern to exploit.  Each misprediction costs ~15–20
cycles on Zen 5, contributing ~2–3 cycles/instruction to CPI.

**Ineffective prefetches**: 5.6 PTI means ~0.56% of all instructions are
prefetches that hit already-cached data.  This is expected and desirable for
Zipf workloads — hot keys stay in L1/L2 and the prefetch is essentially free
(it retires without generating a memory request).  Compare with bench_avx64's
50.7 PTI — the Zipf-distributed lookups mostly hit hot keys already in cache.

## 6. Remaining Bottlenecks

In order of estimated impact:

1. **Branch misprediction (11%)**: The switch dispatch in the mixed loop is the
   primary bottleneck.  Potential mitigations: batch operations by type (loses
   temporal locality), branchless dispatch via function pointer table (adds
   indirect call overhead — see §2.3 caveat), or CMOV-based conditional
   execution (limited by the differing side effects of lookup/insert/delete).

2. **TLB misses (17.7 PTI)**: Addressable with 2MB huge pages
   (`madvise(MADV_HUGEPAGE)` or `mmap` with `MAP_HUGETLB`).  Expected to
   eliminate ~15 PTI of L1 DTLB misses.

3. **L3 refill latency (28.1 PTI)**: At 1M keys, the table exceeds L2 (1 MB)
   but fits in L3 (32 MB).  Prefetching mitigates this effectively.  At larger
   table sizes (>L3), DRAM latency will dominate and prefetch distance may need
   to increase further.

## 7. Files

- `test_del_explore.c` — Four delete variant comparison (vtable dispatch)
- `test_del_h2h.c` — Direct-call head-to-head: backshift vs. tombstone
- `test_pfdist.c` — PF_DIST sweep across 10 values × 5 profiles + dual-distance
- `test_pf_sweep.c` — Hyperparameter sweep: decay × window × threshold × PF pair
- `test_pf_ab.c` — A/B test of adaptive variants (initial, ordering-biased)
- `test_pf_ab2.c` — Controlled A/B: order-rotated, eliminates cache bias

## 8. Methodological Lessons

1. **Cross-session comparisons are unreliable** for <20% differences.
   CPU frequency, cache state, background processes, and NUMA effects
   all vary between runs.  Always use interleaved A/B within the same
   process invocation.

2. **`__attribute__((noinline))` is not the same as separate compilation**,
   but it prevents the most egregious cross-function optimizations.  For
   header-only libraries (like avx_map64.h), the inlined case IS the
   real-world use case.

3. **Execution ordering bias is real**.  The first variant in a sequence
   benefits from warmer cache.  Order-rotation (each variant runs first
   in one round) is essential for fair comparison.

4. **Vtable dispatch artifacts** can dominate micro-benchmark results.
   Function-pointer overhead penalizes code paths with more instructions
   disproportionately.  Always validate with direct/inlined calls.

## References

- Sieve (NSDI '24): "SIEVE is Simpler than LRU" — Yazhuo Zhang et al.
  Simple FIFO with lazy promotion beats ARC/LHD on real workloads.
- AdaptSize (NSDI '17): Berger et al.  Markov chain model adapts cache
  admission parameter.  Inspired our exploration of adaptive parameters,
  though we found a simpler signal (mutation fraction) sufficient.
- The Sieve corollary discovered here: when the optimization parameter is
  constant (or nearly so) across a workload, measurement overhead of ANY
  kind — including trivial integer counters — is wasted.  The parameter
  should be set statically, either at compile time or per call site.
