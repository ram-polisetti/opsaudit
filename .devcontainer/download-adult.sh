#!/usr/bin/env bash
# Idempotent demo-dataset fetch for Codespaces.
#
# Downloads the UCI Adult dataset into the workspace so the synthetic-data
# auditor demo (examples/synthetic-audit-demo/demo.py) can run without
# fetching it at runtime. Safe to re-run: if adult.data already exists,
# nothing is downloaded.
#
# Env:
#   ADULT_DATA  full path where adult.data should live
#               (default: ${containerWorkspaceFolder:-$(pwd)}/.data/adult.data)
set -euo pipefail

TARGET="${ADULT_DATA:-${containerWorkspaceFolder:-$(pwd)}/.data/adult.data}"
URL="https://archive.ics.uci.edu/static/public/2/adult.zip"

if [ -s "$TARGET" ]; then
  echo "adult.data already present at $TARGET - skipping download."
  exit 0
fi

mkdir -p "$(dirname "$TARGET")"
ARCHIVE="$(dirname "$TARGET")/adult.zip"
echo "downloading UCI Adult dataset -> $ARCHIVE ..."
curl -sSL --retry 3 --max-time 300 -o "$ARCHIVE" "$URL"
echo "extracting adult.data ..."
python3 - "$ARCHIVE" "$(dirname "$TARGET")" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as bundle:
    bundle.extract("adult.data", sys.argv[2])
EOF
rm -f "$ARCHIVE"

if [ -s "$TARGET" ]; then
  echo "ready: $TARGET ($(du -h "$TARGET" | cut -f1))"
else
  echo "ERROR: download finished but $TARGET is missing" >&2
  exit 1
fi
