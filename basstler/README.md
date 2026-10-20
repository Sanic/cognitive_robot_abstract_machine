# basstler

The workflow tooling for this repository: the Python behind the stacked-pull-request
tooling, the plan dashboards, the personal-notes hooks and the upstream review reader.

The name is the German word for someone who builds things themselves, and shares its
first letters with the surname of the person who wrote it, Bassiouny.

## Where it lives

`basstler` follows the `src` layout every package in this repository uses: this directory
holds `pyproject.toml` and this README, and the package itself is `src/basstler/`. Nothing
is a module directly in this directory, and a test holds it so.

## Its own environment

**Every session start gives `basstler` an environment of its own, without asking:**
`basstler/.venv`, from an interpreter the package supports, with the package installed
into it editable. Editable, so `import basstler` there is this clone's own source rather
than a second copy of it in `site-packages`. Its dependencies are declared once, in
`pyproject.toml`'s `[project] dependencies`, the way every package in this repository
declares them, and installing the package brings them with it.

The environment comes from `uv` where it is on the path, since `uv` can fetch a supported
interpreter where none is installed, and otherwise from the newest supported `python3.1x`
on the path. A machine with neither - one whose only interpreter is too old - gets `uv`
installed into `basstler/.uv` with the `pip` it has, and `uv` fetches the interpreter.
`uv` also does the installing wherever there is one, since `pip` runs inside the
environment's own interpreter and an older `pip` breaks on a newer one.
`.claude/hooks/session-start.sh` does the work, so no module has to work around a
dependency that is not there, and nobody has to run `pip` by hand.

Three things bound it, and they are worth knowing because it creates an environment and
installs into it:

- **Only for someone who has already set the tooling up.** The hook stops before this
  point when it cannot reach your personal-notes branch, so a clone that has never run
  `/setup-personal-notes` installs nothing at all.
- **Only what is missing.** The usual start finds the environment, looks the declaration
  up and runs no installer, which is what makes doing it every time affordable. The
  package itself counts: an environment without it is reinstalled into.
- **Never fatal.** No `pip` to install `uv` with, or no network: the summary's
  `dependencies:` line says so, naming the command to run, and the rest of the run carries
  on. Callers then fall back to `python3`.

An Actions runner reaches no session hook, so a workflow that runs a module installs the
package itself, editable -
`test/basstler_test/test_package_contract.py` finds every such workflow and checks that it
does.

## Running it

Every entry point is run as a module, by the package's own interpreter:

```bash
basstler/.venv/bin/python -m basstler.stack configuration
basstler/.venv/bin/python -m basstler.maintenance run-report --json
basstler/.venv/bin/python -m basstler.build_dashboard --help
```

The shell entry points under `.claude/hooks/` and the skills that call them all use
`"${BASSTLER_PYTHON}" -m "${SOME_MODULE}"`, with the interpreter and each module named once
in `.claude/hooks/resolve-personal-notes-config.sh`. Sourcing that file also puts
`basstler/src` first on `PYTHONPATH`, so a worktree or a scratch clone runs its own copy.

## What stays outside the package

`SKILL.md` files, `settings.json`, the bash entry points under `.claude/hooks/` and the
plan-dashboard `example/` all stay under `.claude/`, because Claude Code and its readers
find them by path.
