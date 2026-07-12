#!/bin/bash
set -euo pipefail

# Generic personal Claude Code notes hook.
#
# Populates CLAUDE.local.md (gitignored, never committed on any branch) from a
# personal branch on `origin`. No-op in effect for anyone who never creates
# that branch (default or overridden), and collision-free for multiple
# contributors sharing one origin remote if each overrides the branch name via
# local config instead of relying on the shared default.
#
# Works out of the box, zero config: it looks for a branch named
# `claude/personal-notes` on `origin` and, if found, reads
# `.claude/personal/cram-notes.md` off it into CLAUDE.local.md.
# ./create-personal-notes-branch.sh creates that branch (with an empty notes
# file) for anyone who doesn't have it yet.
#
# Override the branch/path per clone, locally (never committed):
#   git config claude.personalNotesBranch <your-branch-name>
#   git config claude.personalNotesPath   <path-on-that-branch>   # optional
#
# git config is per-clone, so it's the wrong mechanism anywhere sessions start
# from a fresh clone every time (e.g. cloud/web sessions) - there's no
# persistent .git/config for it to live in. For that case, override via
# persistent environment variables instead (configured once at the environment
# level, outside the repo, so they survive every fresh clone):
#   CLAUDE_PERSONAL_NOTES_BRANCH=<your-branch-name>
#   CLAUDE_PERSONAL_NOTES_PATH=<path-on-that-branch>   # optional
# See ./README.md for exactly how to wire these into a cloud environment.
# Precedence: git config > environment variable > the `claude/personal-notes`
# default, so a local or environment-level override always wins over it.
#
# Safe to re-run: it only ever overwrites CLAUDE.local.md, and does nothing if
# the configured (or default) branch or path isn't reachable (e.g. a fresh
# clone, or a fork that never created it).
#
# Editing your notes: the written CLAUDE.local.md starts with a short header
# (see below) naming the resolved branch/path and pointing at
# ./save-personal-notes.sh. Since Claude Code loads CLAUDE.local.md as project
# memory every session, that header is always in context - so asking Claude to
# "edit my personal notes" needs no other setup: it edits the file below the
# header, then runs the save script to push the change back.
#
# How this script gets invoked (see ../settings.json): Claude Code registers it
# as a SessionStart hook via `$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh`.
# CLAUDE_PROJECT_DIR is an env var Claude Code itself injects into every hook
# command's environment, resolving to this project's root - so that path is
# correct regardless of Claude Code's own cwd when it runs the hook.
#
# Coexistence with your own settings: Claude Code merges the `hooks` arrays
# across all settings layers (managed > CLI args > .claude/settings.local.json
# > .claude/settings.json (this repo's, committed) > ~/.claude/settings.json)
# by concatenation, not override. So this SessionStart hook runs alongside -
# never instead of - any SessionStart hook you already have configured for
# yourself. settings.json is strict JSON with no comment support, which is why
# this explanation lives here instead of there.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve-personal-notes-config.sh"

git fetch origin "${NOTES_BRANCH}" --quiet 2>/dev/null || exit 0

if git cat-file -e "origin/${NOTES_BRANCH}:${NOTES_PATH}" 2>/dev/null; then
  {
    cat <<HEADER
<!--
Personal notes, synced from '${NOTES_BRANCH}' (${NOTES_PATH}) by session-start.sh.
To edit: change the notes below this line, then run
  "\$CLAUDE_PROJECT_DIR/.claude/hooks/save-personal-notes.sh"
to push the change back to '${NOTES_BRANCH}'. This header is regenerated
every session from git config/environment/default - editing it has no effect.
-->
<!-- END-PERSONAL-NOTES-HEADER -->
HEADER
    git show "origin/${NOTES_BRANCH}:${NOTES_PATH}"
  } > CLAUDE.local.md
fi
