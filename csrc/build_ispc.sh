#!/bin/bash
# CD to the script's directory
cd "$(dirname "$0")"
# This script compiles the ISPC source files into object files and headers.
ispc src/cpu_paged_attention.ispc -o src/cpu_paged_attention_ispc.o -h src/cpu_paged_attention_ispc.h --target=host -O3 --opt=fast-math
