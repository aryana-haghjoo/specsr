#!/usr/bin/env bash
# ci_status.sh — status of the most recent Actions run, with logs if it failed.
#
# The repo is private and git authenticates over SSH, which does not cover the
# REST API. `gh` needs its own token: run `gh auth login` once.
set -uo pipefail
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run:  gh auth login" >&2
  exit 2
fi

echo "=== latest runs on $BRANCH ==="
gh run list --branch "$BRANCH" --limit 5 \
  --json databaseId,name,status,conclusion,headSha,createdAt \
  --template '{{range .}}{{.createdAt}}  {{printf "%-12s" .name}} {{printf "%-12s" .status}} {{printf "%-10s" .conclusion}} {{slice .headSha 0 8}}  {{.databaseId}}{{"\n"}}{{end}}'

id=$(gh run list --branch "$BRANCH" --limit 1 --json databaseId,conclusion \
     --jq 'if .[0].conclusion == "failure" then .[0].databaseId else empty end')
if [ -n "$id" ]; then
  echo
  echo "=== failing steps in run $id ==="
  gh run view "$id" --log-failed 2>/dev/null | head -60
fi
