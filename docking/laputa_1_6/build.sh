#!/usr/bin/env bash
# Build the KANM docking harness against an unmodified Laputa 1.6 source tree.
#
# Compiled Laputa files are those the 1.6 VC++ and Xcode projects compile
# (Amount.cpp is not in either project; Amount.h is header-only).
#
# Usage: docking/laputa_1_6/build.sh /path/to/Laputa-1.6 [build_dir]
#
# Laputa 1.6 (GPLv3) is not distributed with KANM. Obtain it from the
# SourceForge project "epistemenet" (https://sourceforge.net/projects/epistemenet/).
# Every Laputa file used is checked against laputa_1_6_sources.sha256 before
# compiling, so the build fails if any model file differs from the archive.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:?usage: build.sh /path/to/Laputa-1.6 [build_dir]}"
OUT_INPUT="${2:-$HERE/build}"

# Refuse to replace any caller-owned path. Build in a fresh sibling directory
# and publish it only after compilation succeeds; cleanup is limited to the
# unique staging directory created below.
case "$OUT_INPUT" in
  /*) ;;
  *) OUT_INPUT="$PWD/$OUT_INPUT" ;;
esac
OUT_PARENT="$(dirname "$OUT_INPUT")"
OUT_NAME="$(basename "$OUT_INPUT")"
if [[ "$OUT_NAME" == "/" || "$OUT_NAME" == "." || "$OUT_NAME" == ".." ]]; then
  echo "Build output must name a new directory, not a filesystem root or dot path." >&2
  exit 1
fi
mkdir -p "$OUT_PARENT"
OUT_PARENT="$(cd "$OUT_PARENT" && pwd)"
OUT="$OUT_PARENT/$OUT_NAME"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "Output path already exists; choose a new build_dir (it will not be deleted): $OUT" >&2
  exit 1
fi

SRC="$(cd "$SRC" && pwd)"
if [[ "$OUT/" == "$SRC/"* || "$SRC/" == "$OUT/"* ]]; then
  echo "Output path and Laputa source tree must not contain one another." >&2
  exit 1
fi

(cd "$SRC" && sha256sum --check --quiet "$HERE/laputa_1_6_sources.sha256") || {
  echo "Laputa source does not match the recorded 1.6 archive checksums." >&2; exit 1; }

STAGE="$(mktemp -d "$OUT.stage.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT
mkdir -p "$STAGE/src" "$STAGE/headers" "$STAGE/obj"
cp "$SRC"/src/{Society,Inquirer,Link,Trust,Distribution,MetaDistribution,Parameters,StatisticsBlock}.cpp "$STAGE/src/"
cp "$SRC"/headers/*.h "$SRC"/headers/*.tcc "$STAGE/headers/"
rm "$STAGE/headers/App.h"          # GUI application header, replaced by stubs/App.h
cp "$SRC"/lib/tinyxml/*.cpp "$SRC"/lib/tinyxml/*.h "$STAGE/src/"
mkdir -p "$STAGE/distributions"; cp "$SRC"/build/data/distributions/*.distr "$STAGE/distributions/"

CXX="${CXX:-g++}"
FLAGS=(-std=c++11 -O2 -w -I"$HERE/stubs" -I"$STAGE/headers" -I"$STAGE/src"
  "-DMEMORY_ALLOCATOR_LINK=std::allocator<std::pair<const unsigned int, Link> >"
  "-DMEMORY_ALLOCATOR_INT=std::allocator<int>"
  "-DMEMORY_ALLOCATOR_FLOAT=std::allocator<float>"
  '-DDATA_PATH="data/"' '-DDOCS_PATH="docs/"')
for f in "$STAGE"/src/*.cpp "$HERE/harness.cpp"; do
  "$CXX" "${FLAGS[@]}" -c "$f" -o "$STAGE/obj/$(basename "${f%.cpp}").o"
done
"$CXX" "$STAGE"/obj/*.o -o "$STAGE/laputa16_harness" -Wl,--unresolved-symbols=report-all
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "Output path appeared during build; refusing to replace it: $OUT" >&2
  exit 1
fi
mv "$STAGE" "$OUT"
trap - EXIT
echo "built $OUT/laputa16_harness"
