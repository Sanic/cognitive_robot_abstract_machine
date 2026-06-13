# Python Role Lens (PyCharm plugin)

Adds IDE awareness for the **Role / delegation** pattern in Python — the same support a
companion VS Code extension provides — so completion, type inference, and Go‑to‑Declaration
work on delegated attributes.

```python
@dataclass
class Role(Generic[RoleTakerT]):
    _taker: RoleTakerT
    def __getattr__(self, name: str) -> Any:
        return getattr(self._taker, name)

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Teacher(Role[Person]):
    courses: list[str]

teacher = Teacher(_taker=Person("Ahmed", 20))
teacher.name   # ← autocompletes, infers `str`, and Ctrl/Cmd+Click jumps to Person.name
```

Because access is delegated through `__getattr__` (composition, not inheritance), PyCharm
normally sees nothing on `teacher` except `courses`. This plugin injects the taker's members
onto any `Role[…]` subclass.

---

## What you get

A single extension point — a **class members provider** — drives all of this at once,
because each injected member resolves to the *real* declaration in the taker class:

| Feature | How it works |
|---|---|
| Code completion after `.` | Injected members appear in the completion list |
| Type inference (`teacher.name` → `str`) | Type is read from the member's resolve target |
| Go to Declaration (Ctrl/Cmd+Click) | Navigates to the attribute/method in the taker |
| Find Usages / Rename | Operate on the real target element |

### Delegation cases handled

- **Direct roles** — `Teacher(Role[Person])`.
- **Transitive roles** — `SeniorTeacher(Teacher)` still sees `Person`'s members (the
  `Role[…]` base is searched across the whole ancestry).
- **Inherited taker members** — if `Person(Entity)`, `Entity`'s members are delegated too.
- **Nested roles** — if the taker is itself a role (`Department(Role[Teacher])`), its
  delegated members are pulled in recursively.

---

## Project layout

```
pyroles-pycharm/
├─ build.gradle.kts                  # IntelliJ Platform Gradle Plugin 2.x config
├─ settings.gradle.kts               # plugin management + repositories
├─ gradle.properties                 # plugin coordinates & target PyCharm version
├─ gradlew / gradlew.bat             # Gradle wrapper (Gradle 9.5)
├─ gradle/wrapper/…
├─ sample/
│  └─ roles_demo.py                  # open this in `runIde` to try it out
└─ src/main/
   ├─ kotlin/com/example/pyroles/
   │  └─ RoleMembersProvider.kt      # the whole implementation
   └─ resources/META-INF/
      ├─ plugin.xml                  # registers the provider
      └─ pluginIcon.svg
```

---

## Prerequisites

- **JDK 21** (the Gradle toolchain will auto-provision it if missing).
- Internet access on first build (Gradle, the IntelliJ Platform, and PyCharm are downloaded
  and cached).
- IntelliJ IDEA is recommended for editing, but not required to build.

---

## Configure it for your project

There is **one** thing worth setting and two optional knobs, all at the top of
`RoleMembersProvider.kt`:

```kotlin
// The fully-qualified name of YOUR Role base class.
// e.g. `from myapp.roles import Role`  ->  "myapp.roles.Role"
const val ROLE_QUALIFIED_NAME: String = "roles.Role"

// Change these only if your delegation field or method differ from the norm.
const val DELEGATION_FIELD: String = "_taker"
const val GETATTR: String = "__getattr__"
```

`ROLE_QUALIFIED_NAME` is a fast, precise match. **You can also leave it as-is:** when it
doesn't match, the plugin falls back to a *structural* check — any base class that declares
a `_taker` field and a `__getattr__` method is treated as a role. That fallback is what lets
the bundled `sample/roles_demo.py` work without configuration, and it handles codebases with
several Role-like bases.

---

## Build, run, install

From the project root:

```bash
# Launch a sandbox PyCharm with the plugin loaded (best way to develop/test)
./gradlew runIde

# Build the installable plugin zip -> build/distributions/pyroles-pycharm-<version>.zip
./gradlew buildPlugin

# Verify compatibility / catch common plugin mistakes
./gradlew verifyPlugin
```

To install the built zip in your own PyCharm:
**Settings → Plugins → ⚙ → Install Plugin from Disk…** → pick the zip from
`build/distributions/`, then restart.

### Trying it out

In the sandbox IDE from `runIde`, open `sample/roles_demo.py` and:

- type a `.` after `teacher`, `senior`, or `dept` and watch delegated members appear;
- hover a delegated attribute to see its inferred type;
- Ctrl/Cmd+Click a delegated attribute to jump to its declaration.

---

## How it works (internals)

PyCharm has its own type/resolve engine; you extend it rather than re-implementing editor
features. The plugin registers a `Pythonid.pyClassMembersProvider` whose
`getMembers(PyClassType, PsiElement, TypeEvalContext)` returns synthetic
[`PyCustomMember`]s for the taker's attributes and methods.

The algorithm (see `RoleMembersProvider.kt`):

1. Skip class objects (`Teacher.x`); only **instances** delegate.
2. Find the taker(s): scan the role class **and its ancestors** for `Role[X]` base
   expressions, resolving `X` to a `PyClass`. (handles *direct* + *transitive* roles)
3. For each taker, walk its **full MRO** — minus `Role`, `object`, `Generic` — collecting
   class-level fields, instance attributes, and methods. (handles *inherited* members)
4. If a taker is itself a role, recurse into it. (handles *nested* roles)
5. De-duplicate by name, closest-in-MRO wins; drop dunders and the `_taker` field.

Each `PyCustomMember` is created with the real target element, so navigation and type
inference come for free — no separate definition or completion contributor is needed.

---

## Compatibility notes & fallbacks

The Python plugin API is stable but JetBrains occasionally moves classes or tweaks
signatures between major releases. If something doesn't resolve in your target version:

- **An import is red.** `PyClassMembersProviderBase` / `PyCustomMember` are sometimes found
  under `com.jetbrains.python.codeInsight` and sometimes under `com.jetbrains.python.psi.types`.
  Press **Alt+Enter → Import** and let the IDE pick the right package — the class names are
  stable, only the package occasionally moves.
- **`getMembers` won't override.** Very old platforms used a 2-argument form without
  `TypeEvalContext`. Match the signature your SDK declares (Ctrl+B into the base class).
- **Completion works but Ctrl+Click doesn't.** The base class resolves navigation by
  matching the members from `getMembers`. If your version behaves differently, override
  `resolveMember(...)` to return the target of the matching member.

### Known limitations

- The **constructor is not augmented** — PyCharm still sees `Teacher(_taker=…, courses=…)`,
  not `Teacher(name=…, age=…)`. Augmenting synthesized `__init__` parameters from a plugin
  is not currently feasible. This affects construction only, never attribute access.
- **String forward references** in the base (`Role["Person"]`) are not resolved; use a
  direct name reference.

### Alternative if the Gradle setup fights you

Versions here are pinned to a known-good, mutually compatible set (Gradle 9.5, IntelliJ
Platform Gradle Plugin 2.16.0, Kotlin 2.1.20, target PyCharm 2024.3). If you hit Gradle/DSL
friction on a different machine, the most robust path is to start from the official
[IntelliJ Platform Plugin Template](https://github.com/JetBrains/intellij-platform-plugin-template)
and transplant just two things into it:

1. `src/main/kotlin/com/example/pyroles/RoleMembersProvider.kt`
2. the `<depends>` + `<extensions>` block from `plugin.xml`

then add `bundledPlugin("PythonCore")` to its `intellijPlatform { … }` dependencies.

---

## License

Provided as-is, no warranty. Adapt freely.
