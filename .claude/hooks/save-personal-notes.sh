#!/bin/bash
set -euo pipefail

# Persists edits made to CLAUDE.local.md back onto the personal-notes branch
# on origin, so the next session's session-start.sh reads the updated notes.
# This is the write half of the loop session-start.sh's own header
# (see ./session-start.sh) points Claude at when asked to edit your notes:
# edit CLAUDE.local.md below its header, then run this script.
#
# Usage (from the project root, after editing CLAUDE.local.md):
#   "$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
#
# Resolves the branch/path exactly like session-start.sh (git config >
# environment variable > the `claude/personal-notes` default), so it always
# writes back to wherever the notes were read from.
#
# Strips the auto-generated sync header (everything up to and including the
# "END-PERSONAL-NOTES-HEADER" marker line) before pushing, so the header
# itself never ends up committed to the notes branch - only your actual notes
# content does.
#
# Safe to re-run: a no-op if CLAUDE.local.md's content (after stripping the
# header) already matches what's on the branch. Does its work in a scratch
# worktree, so it never touches your current branch or working tree. Fails
# with a clear message if the target branch doesn't exist yet on origin - run
# ./create-personal-notes-branch.sh first in that case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

if [ ! -f CLAUDE.local.md ]; then
  echo "No CLAUDE.local.md in the current directory - nothing to save." >&2
  exit 1
fi

if ! git fetch origin "${NOTES_BRANCH}" --quiet 2>/dev/null; then
  echo "Branch '${NOTES_BRANCH}' doesn't exist on origin yet." >&2
  echo "Run ./create-personal-notes-branch.sh first, then re-run this script." >&2
  exit 1
fi

CONTENT_FILE="$(mktemp)"
SCRATCH_DIR="$(mktemp -d)"
cleanup() {
  git worktree remove --force "${SCRATCH_DIR}" 2>/dev/null || rm -rf "${SCRATCH_DIR}"
  git branch -D __save-personal-notes-tmp > /dev/null 2>&1 || true
  rm -f "${CONTENT_FILE}"
}
trap cleanup EXIT

CONTENT_START="$(grep -n '^<!-- END-PERSONAL-NOTES-HEADER -->$' CLAUDE.local.md | head -1 | cut -d: -f1)"
if [ -n "${CONTENT_START}" ]; then
  tail -n "+$((CONTENT_START + 1))" CLAUDE.local.md > "${CONTENT_FILE}"
else
  cp CLAUDE.local.md "${CONTENT_FILE}"
fi

git branch -D __save-personal-notes-tmp > /dev/null 2>&1 || true
git worktree add -b __save-personal-notes-tmp "${SCRATCH_DIR}" "origin/${NOTES_BRANCH}" --quiet

mkdir -p "${SCRATCH_DIR}/$(dirname "${NOTES_PATH}")"
cp "${CONTENT_FILE}" "${SCRATCH_DIR}/${NOTES_PATH}"
git -C "${SCRATCH_DIR}" add "${NOTES_PATH}"

if git -C "${SCRATCH_DIR}" diff --cached --quiet -- "${NOTES_PATH}"; then
  echo "No changes to save - '${NOTES_PATH}' on '${NOTES_BRANCH}' is already up to date."
  exit 0
fi

git -C "${SCRATCH_DIR}" commit --quiet -m "Update personal notes"
git -C "${SCRATCH_DIR}" push origin "HEAD:${NOTES_BRANCH}"

echo "Saved '${NOTES_PATH}' back to '${NOTES_BRANCH}'."
