CC      ?= gcc

BIN = bin

# Aggressive whole-program optimization flags
OPTFLAGS = -O3 -march=native -flto \
           -fomit-frame-pointer -fno-asynchronous-unwind-tables \
           -fno-unwind-tables -fno-stack-protector \
           -fmerge-all-constants -DNDEBUG

CFLAGS  ?= $(OPTFLAGS) -Wall -Wextra -std=gnu11
LDFLAGS ?= -lgit2 -ldw -lelf -lcapstone -lpfm -lm

# Vendor paths
TS_DIR  = vendor/tree-sitter
TSC_DIR = vendor/tree-sitter-c

# Include paths (tree-sitter/lib/src needed for lib.c unity build internals)
INCLUDES = -I $(TS_DIR)/lib/include -I $(TS_DIR)/lib/src -I $(TSC_DIR)/src -I hmap

# All sources compiled in one invocation — gives the compiler full visibility
# across tree-sitter internals, the grammar, and archmap for maximum inlining.
SRCS = main.c skeleton.c output.c \
       perf_analysis.c arena.c symres.c perf_tools.c perf_report.c \
       git_cache.c \
       $(TS_DIR)/lib/src/lib.c $(TSC_DIR)/src/parser.c

TARGET = $(BIN)/archmap

.PHONY: all clean vendor-clean

all: $(TARGET)

$(BIN):
	mkdir -p $@

# --- Vendor fetch ---

$(TS_DIR):
	mkdir -p vendor
	curl -sL https://github.com/tree-sitter/tree-sitter/archive/refs/tags/v0.26.5.tar.gz \
		| tar xz -C vendor
	mv vendor/tree-sitter-0.26.5 $@

$(TSC_DIR):
	mkdir -p vendor
	mkdir -p $@
	curl -sL https://github.com/tree-sitter/tree-sitter-c/releases/download/v0.24.1/tree-sitter-c.tar.gz \
		| tar xz -C $@

# --- Single whole-program compilation ---

$(TARGET): main.c skeleton.c skeleton.h output.c output.h \
           perf_analysis.c perf_analysis.h arena.c arena.h symres.c symres.h \
           perf_tools.c perf_tools.h perf_report.c perf_report.h \
           git_cache.c git_cache.h hmap/hmap_shim.h | $(BIN) $(TS_DIR) $(TSC_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) -o $@ $(SRCS)

BENCH_CFLAGS  = -O3 -march=native -std=gnu11
BENCH_CXXFLAGS = -O3 -march=native -std=gnu++17

$(BIN)/test_hashmap: hmap/test_hashmap_main.cpp hmap/test_hashmap.c hmap/avx_map64s.h hmap/simd_map64.h hmap/verstable.h | $(BIN)
	$(CC) $(BENCH_CFLAGS) -c -o $(BIN)/test_hashmap_bench.o hmap/test_hashmap.c
	g++ $(BENCH_CXXFLAGS) -c -o $(BIN)/test_hashmap_main.o hmap/test_hashmap_main.cpp
	g++ $(BENCH_CXXFLAGS) -o $@ $(BIN)/test_hashmap_main.o $(BIN)/test_hashmap_bench.o -lm

$(BIN)/test_hashmap_512: hmap/test_hashmap_main.cpp hmap/test_hashmap.c hmap/avx_map64s.h hmap/simd_map64.h hmap/verstable.h | $(BIN)
	$(CC) $(BENCH_CFLAGS) -mavx512f -mavx512bw -c -o $(BIN)/test_hashmap_bench.o hmap/test_hashmap.c
	g++ $(BENCH_CXXFLAGS) -mavx512f -mavx512bw -c -o $(BIN)/test_hashmap_main.o hmap/test_hashmap_main.cpp
	g++ $(BENCH_CXXFLAGS) -mavx512f -mavx512bw -o $@ $(BIN)/test_hashmap_main.o $(BIN)/test_hashmap_bench.o -lm

$(BIN)/bench_512: hmap/bench_backend.c hmap/simd_map64.h | $(BIN)
	$(CC) $(BENCH_CFLAGS) -mavx512f -mavx512bw -o $@ hmap/bench_backend.c -lm

$(BIN)/bench_avx2: hmap/bench_backend.c hmap/simd_map64.h | $(BIN)
	$(CC) $(BENCH_CFLAGS) -mno-avx512f -mno-avx512bw -o $@ hmap/bench_backend.c -lm

$(BIN)/bench_scalar: hmap/bench_backend.c hmap/simd_map64.h | $(BIN)
	$(CC) $(BENCH_CFLAGS) -mno-avx512f -mno-avx512bw -mno-avx2 -mno-sse4.2 -o $@ hmap/bench_backend.c -lm

bench_compare: $(BIN)/bench_512 $(BIN)/bench_avx2 $(BIN)/bench_scalar
	@echo "=== AVX-512 ===" && ./$(BIN)/bench_512 && echo && echo "=== AVX2 ===" && ./$(BIN)/bench_avx2 && echo && echo "=== Scalar ===" && ./$(BIN)/bench_scalar

clean:
	rm -rf $(BIN)

vendor-clean:
	rm -rf vendor
