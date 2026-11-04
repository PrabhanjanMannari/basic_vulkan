#!/usr/bin/env bash
set -e

SLANGC="/home/prabhanjan/vulkansdk-linux-x86_64-1.4.363.0/1.4.363.0/x86_64/bin/slangc"  # Modify if necessary

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <input.slang> <output.spv> -entry <entry1> [-entry <entry2> ...]"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

"$SLANGC" "$INPUT" \
    -target spirv \
    -profile spirv_1_4 \
    -emit-spirv-directly \
    -fvk-use-entrypoint-name \
    "$@" \
    -o "$OUTPUT"
