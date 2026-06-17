#!/usr/bin/env bash
# Serve the repo over HTTP and open the capability-tree viewer, which then
# auto-loads the DRChallenge tree + drtulu results (file:// can't fetch them).
#
# Run from the repo root:
#   bash EvalTree/serve_viewer.sh
set -euo pipefail

PORT="${1:-8000}"
URL="http://localhost:${PORT}/EvalTree/tree_viewer.html"

# Serve from the repo root so the viewer's root-relative /Datasets/... paths resolve.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "==> Serving ${ROOT} at http://localhost:${PORT}"
echo "==> Opening ${URL}"

python3 -m http.server "$PORT" --directory "$ROOT" >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

sleep 1
( open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || echo "Open $URL in your browser." )

echo "==> Press Ctrl-C to stop the server."
wait "$SERVER_PID"
