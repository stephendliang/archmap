# archmap improvement ideas

From real use profiling an HPACK codec (hpack.c, 830 LOC → 100 LOC archmap output, 8× compression).

## What works

- Function + call-graph edges (`→`) give full hot-path trace without reading source
- Cross-file `-d` resolves types/defines/data across headers and generated .inc files
- Line number annotations (`//:122-146`) allow targeted source reads
- Token-efficient: exactly what an LLM needs to reason about architecture

## Missing

**`--sizes`** — Data/struct byte sizes. `huff_accel[518][16]` is 33KB (blows L1d) but archmap just says `huff_accel[518][16] = { ... }`. Show `(33,152 B)`. Same for structs: `hpack_ctx (9,232 B)` vs `huff_accel_entry (4 B, packed)`.

**`--branch-stats`** — Per-function branch/loop counts. `swar_memcpy_short` has 6 conditional branches in a cascade — flag this as `[6 br, 0 loops]`. Identifies branch-pressure hotspots before profiling.

**`--expand-inline`** — Show `always_inline` functions as nested call sites at the expansion point rather than just an edge. Maps to what profilers actually report.

**`--profile FILE`** — Ingest perf/AMDuProf CSV, annotate call graph with cycle counts. One view instead of cross-referencing two tools.

**`--diff OLD NEW`** — Show added/removed/changed functions and call edges. Perfect for reviewing changes without reading full diffs.

**`--hot-path FUNC`** — Trace all transitive callees from a function, pruning unrelated code. `--hot-path hpack_decode` would show only the decode subtree.
