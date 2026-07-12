# Sourced (not executed) by session-start.sh, create-personal-notes-branch.sh
# and save-personal-notes.sh, so all three resolve the personal-notes branch
# and path with the exact same precedence: git config > environment variable
# > the `claude/personal-notes` zero-config default. See ./README.md.

NOTES_BRANCH="$(git config --get claude.personalNotesBranch || true)"
NOTES_BRANCH="${NOTES_BRANCH:-${CLAUDE_PERSONAL_NOTES_BRANCH:-claude/personal-notes}}"

NOTES_PATH="$(git config --get claude.personalNotesPath || true)"
NOTES_PATH="${NOTES_PATH:-${CLAUDE_PERSONAL_NOTES_PATH:-.claude/personal/cram-notes.md}}"
