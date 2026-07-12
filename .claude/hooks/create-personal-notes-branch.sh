#!/bin/bash
set -euo pipefail

# One-time helper: creates the personal-notes branch (default:
# `claude/personal-notes`) on `origin` with a single empty notes file at the
# default path (`.claude/personal/cram-notes.md`), and pushes it - so
# session-start.sh has something to read out of the box, with no config
# needed on your end.
#
# Usage (from anywhere inside the repo):
#   ./.claude/hooks/create-personal-notes-branch.sh
#
# Resolves the branch/path the same way session-start.sh does (git config >
# environment variable > default), so it always targets whatever a fresh
# session would actually read from. Override via the same two variable names
# it and configure-personal-notes.sh read:
#   CLAUDE_PERSONAL_NOTES_BRANCH=<branch> CLAUDE_PERSONAL_NOTES_PATH=<path> \
#     ./.claude/hooks/create-personal-notes-branch.sh
#
# Safe to re-run: refuses to touch a branch that already exists locally or on
# origin, so it never overwrites existing notes. Does its work in a scratch
# worktree, so it never touches your current branch or working tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if git show-ref --verify --quiet "refs/heads/${NOTES_BRANCH}"; then
  echo "Branch '${NOTES_BRANCH}' already exists locally - not touching it." >&2
  exit 1
fi

if git ls-remote --exit-code --heads origin "${NOTES_BRANCH}" > /dev/null 2>&1; then
  echo "Branch '${NOTES_BRANCH}' already exists on origin - not touching it." >&2
  exit 1
fi

SCRATCH_DIR="$(mktemp -d)"
trap 'git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"' EXIT

git worktree add --orphan -b "${NOTES_BRANCH}" "${SCRATCH_DIR}" --quiet
git -C "${SCRATCH_DIR}" rm -rf --quiet . > /dev/null 2>&1 || true

mkdir -p "${SCRATCH_DIR}/$(dirname "${NOTES_PATH}")"
: > "${SCRATCH_DIR}/${NOTES_PATH}"

git -C "${SCRATCH_DIR}" add "${NOTES_PATH}"
git -C "${SCRATCH_DIR}" commit --quiet -m "Initialize personal notes branch"
git -C "${SCRATCH_DIR}" push origin "${NOTES_BRANCH}"

git worktree remove --force "${SCRATCH_DIR}"
git branch -D "${NOTES_BRANCH}" > /dev/null
trap - EXIT

echo "Created and pushed '${NOTES_BRANCH}' with an empty '${NOTES_PATH}'."
echo "Your current branch and working tree are untouched."
