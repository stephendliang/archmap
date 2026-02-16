CC      ?= gcc

# Aggressive whole-program optimization flags
OPTFLAGS = -O3 -march=native -flto \
           -fomit-frame-pointer -fno-asynchronous-unwind-tables \
           -fno-unwind-tables -fno-stack-protector \
           -fipa-pta -fmerge-all-constants -DNDEBUG \
           -mavx512f -mavx512bw

CFLAGS  ?= $(OPTFLAGS) -Wall -Wextra -std=gnu11
LDFLAGS ?= -lgit2

# Vendor paths
TS_DIR  = vendor/tree-sitter
TSC_DIR = vendor/tree-sitter-c

# Include paths (tree-sitter/lib/src needed for lib.c unity build internals)
INCLUDES = -I $(TS_DIR)/lib/include -I $(TS_DIR)/lib/src -I $(TSC_DIR)/src

# All sources compiled in one invocation — gives the compiler full visibility
# across tree-sitter internals, the grammar, and archmap for maximum inlining.
SRCS = main.c perf_analysis.c git_cache.c hmap_avx512.c $(TS_DIR)/lib/src/lib.c $(TSC_DIR)/src/parser.c

TARGET = archmap

.PHONY: all clean vendor-clean

all: $(TARGET)

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

$(TARGET): main.c perf_analysis.c perf_analysis.h git_cache.c git_cache.h hmap_avx512.c hmap_avx512.h | $(TS_DIR) $(TSC_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) -o $@ $(SRCS)

BENCH_CFLAGS  = -O3 -march=native -mavx512f -mavx512bw -std=gnu11
BENCH_CXXFLAGS = -O3 -march=native -mavx512f -mavx512bw -std=gnu++17

test_hashmap: test_hashmap_main.cpp test_hashmap.c avx_map64s.h avx_map64.h verstable.h
	$(CC) $(BENCH_CFLAGS) -c -o test_hashmap_bench.o test_hashmap.c
	g++ $(BENCH_CXXFLAGS) -c -o test_hashmap_main.o test_hashmap_main.cpp
	g++ $(BENCH_CXXFLAGS) -o $@ test_hashmap_main.o test_hashmap_bench.o -lm

clean:
	rm -f $(TARGET) test_hashmap test_hashmap_c.o test_hashmap_cpp.o

vendor-clean:
	rm -rf vendor
