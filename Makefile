CC      ?= gcc

# Aggressive whole-program optimization flags
OPTFLAGS = -O3 -march=native -flto -fwhole-program \
           -fomit-frame-pointer -fno-asynchronous-unwind-tables \
           -fno-unwind-tables -fno-stack-protector \
           -fipa-pta -fmerge-all-constants -DNDEBUG

CFLAGS  ?= $(OPTFLAGS) -Wall -Wextra -std=gnu11
LDFLAGS ?=

# Vendor paths
TS_DIR  = vendor/tree-sitter
TSC_DIR = vendor/tree-sitter-c

# Include paths (tree-sitter/lib/src needed for lib.c unity build internals)
INCLUDES = -I $(TS_DIR)/lib/include -I $(TS_DIR)/lib/src -I $(TSC_DIR)/src

# All sources compiled in one invocation — gives the compiler full visibility
# across tree-sitter internals, the grammar, and archmap for maximum inlining.
SRCS = main.c $(TS_DIR)/lib/src/lib.c $(TSC_DIR)/src/parser.c

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

$(TARGET): main.c | $(TS_DIR) $(TSC_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

vendor-clean:
	rm -rf vendor
