#!/bin/bash
# fetch.sh — download C projects for archmap testing
set -euo pipefail
cd "$(dirname "$0")"

fetch_tar() {
    local name="$1" url="$2" strip="$3"
    if [ -d "$name" ]; then
        echo "skip: $name (already exists)"
        return
    fi
    echo "fetch: $name"
    mkdir -p "$name"
    curl -fSL "$url" | tar xz --strip-components="$strip" -C "$name"
}

fetch_git() {
    local name="$1" url="$2"
    if [ -d "$name" ]; then
        echo "skip: $name (already exists)"
        return
    fi
    echo "clone: $name"
    git clone --depth 1 --recursive "$url" "$name"
}

# nginx — large, real-world, deeply nested
fetch_tar nginx \
    "https://github.com/nginx/nginx/archive/refs/tags/release-1.29.5.tar.gz" 1

# khashl — tiny single-header hash table (stress test: minimal input)
fetch_git khashl \
    "https://github.com/attractivechaos/khashl.git"

# CC — header-only generic containers (macro-heavy, stress test for preprocessor)
fetch_tar cc \
    "https://github.com/JacksonAllan/CC/archive/refs/tags/v1.4.3.tar.gz" 1

# libreactor — small async I/O library (clean modern C, submodules have core types)
fetch_git libreactor \
    "https://github.com/fredrikwidlund/libreactor.git"

echo "done. projects:"
for d in */; do
    count=$(find "$d" -name '*.c' -o -name '*.h' | wc -l)
    printf "  %-14s %4d files\n" "${d%/}" "$count"
done
