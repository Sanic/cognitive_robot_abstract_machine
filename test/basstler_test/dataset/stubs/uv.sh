#!/bin/bash
set -uo pipefail

# Test stub standing in for `uv`, so session-start.sh's tests can exercise the
# environment it creates without fetching or creating a real interpreter.
# Copied into place as an executable named `uv`, earlier on PATH than any real
# one; see the stub_bin fixture in conftest.py.
#
# Recognizes only the two invocations the hooks make:
#   `venv --no-project --python <requirement> <directory>`, from
#     create_basstler_environment, which it answers by writing
#     <directory>/bin/python as a script running STUB_UV_PYTHON - the
#     interpreter running the suite;
#   `pip install --python <interpreter> --editable <source tree>`, from
#     install_basstler_into_environment, which installs nothing.
#   STUB_UV_CALL_LOG - file the invocation is appended to
#   STUB_UV_STATUS   - exit status to report, default 0; anything else creates
#                      and installs nothing
#
# Exits 64 on an unrecognized invocation for the same reason gh.sh does: a
# changed call must fail a test rather than pass by accident.

if [ -n "${STUB_UV_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "${STUB_UV_CALL_LOG}"
fi

STATUS="${STUB_UV_STATUS:-0}"

if [ "${1:-}" = "pip" ] && [ "${2:-}" = "install" ] && [ "${3:-}" = "--python" ] \
    && [ -n "${4:-}" ] && [ "${5:-}" = "--editable" ] && [ -n "${6:-}" ]; then
  if [ "${STATUS}" != "0" ]; then
    echo "error: stub uv refused to install $6" >&2
    exit "${STATUS}"
  fi
  echo "Installed $6"
  exit 0
fi

if [ "${1:-}" != "venv" ] || [ "${2:-}" != "--no-project" ] \
    || [ "${3:-}" != "--python" ] || [ -z "${4:-}" ] || [ -z "${5:-}" ]; then
  echo "uv stub: unrecognized invocation: $*" >&2
  exit 64
fi

if [ "${STATUS}" != "0" ]; then
  echo "error: stub uv found no interpreter satisfying $4" >&2
  exit "${STATUS}"
fi

mkdir -p "$5/bin"
printf '#!/bin/sh\nexec "%s" "$@"\n' "${STUB_UV_PYTHON}" > "$5/bin/python"
chmod +x "$5/bin/python"
echo "Creating virtual environment at: $5"
