#!/bin/bash
set -uo pipefail

# Test stub standing in for `pip`, so session-start.sh's tests can exercise
# the install it runs without installing anything into the machine running
# them. Copied into place as an executable named `pip`, earlier on PATH than
# the real one; see the stub_bin fixture in conftest.py.
#
# Recognizes only the two invocations the hooks make:
#   `--python <interpreter> install --editable <source tree>`, from
#     install_basstler_into_environment;
#   `install --target <directory> uv`, from install_uv, which it answers by
#     writing <directory>/bin/uv as a script running bash on
#     STUB_PIP_INSTALLED_UV.
#   STUB_PIP_CALL_LOG    - file the invocation is appended to
#   STUB_PIP_STATUS      - exit status to report, default 0; anything else
#                          installs nothing
#   STUB_PIP_INSTALLED_UV - the script an installed uv runs
#
# Exits 64 on an unrecognized invocation for the same reason gh.sh does: a
# changed call must fail a test rather than pass by accident.

if [ -n "${STUB_PIP_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "${STUB_PIP_CALL_LOG}"
fi

STATUS="${STUB_PIP_STATUS:-0}"

if [ "${1:-}" = "install" ] && [ "${2:-}" = "--target" ] && [ -n "${3:-}" ] \
    && [ "${4:-}" = "uv" ]; then
  if [ "${STATUS}" != "0" ]; then
    echo "ERROR: stub pip refused to install uv" >&2
    exit "${STATUS}"
  fi
  mkdir -p "$3/bin"
  printf '#!/bin/sh\nexec bash "%s" "$@"\n' "${STUB_PIP_INSTALLED_UV}" > "$3/bin/uv"
  chmod +x "$3/bin/uv"
  echo "Successfully installed uv"
  exit 0
fi

if [ "${1:-}" != "--python" ] || [ -z "${2:-}" ] || [ "${3:-}" != "install" ] \
    || [ "${4:-}" != "--editable" ] || [ -z "${5:-}" ]; then
  echo "pip stub: unrecognized invocation: $*" >&2
  exit 64
fi

if [ "${STATUS}" = "0" ]; then
  echo "Successfully installed $5"
  exit 0
fi
echo "ERROR: stub pip refused to install $5" >&2
exit "${STATUS}"
