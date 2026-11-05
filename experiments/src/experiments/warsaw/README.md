# The Warsaw pipeline

A labelled scan of a room goes in; an annotated, hierarchical world comes out. The run
asks a vision-language model which class each of the scan's labels means, who owns the
faces two labels both claim, and what each body is, then cuts the scan into one body per
object and mounts the parts into their wholes.

    python -m experiments.warsaw.pipeline.pipeline

That runs the settings as they stand. Nothing is passed on a command line: everything a
run can be told is a field of `PipelineSettings` in `pipeline/settings.py`, and a run
writes every one of them to its `provenance.json`, so a run can be read back from its own
directory rather than from someone's shell history. How to run another version of the
pipeline -- another model, without pictures, another way of proposing classes -- is in
[Running a version of the pipeline](#running-a-version-of-the-pipeline).

On some machines the renders come back blank and the run stops with a
`BlankRenderError`; see below for what to do about that.

`ASKING_ABOUT_PHOTOGRAPHS.md` records an idea that is not built: showing a model the
source photographs with the faces in question drawn over them, rather than renders of a
scan whose handles are nineteen triangles.

## Where the scan goes

    experiments/src/experiments/warsaw/dataset/<scene name>/<anything>.ply

One directory per scene, holding exactly one `.ply`. `WarsawScene.from_directory` globs
`*.ply` and refuses a directory with none or with more than one, so a second mesh beside
the scene is an error rather than a coin toss.

`dataset/` is not in the repository — it is gitignored, along with `pipeline_runs/` — so
a fresh clone has no such directory and you make it yourself. Which scene is read is
`PipelineSettings.scene_directory`, whose default is
`dataset/kitchen2_meshes_out_20260910`.

## What the scan has to be

A single mesh whose faces carry the labels, written as **one integer face property per
class, named for the class**:

    element face 1749164
    property list uchar int vertex_indices
    property int floor
    property int cabinet
    property int drawer
    ...

The value of a face's `cabinet` property is *which* cabinet the face belongs to; `0`
means no cabinet covers it. So one property per class, one instance number per face, and
a face may belong to objects of several classes at once — a drawer front is part of the
drawer and of the cabinet holding it. Sorting that out is what the pipeline does.

Two things about the file are easy to lose:

- **Only the PLY reader keeps these properties**, and only when the mesh is loaded
  unprocessed. Re-exporting the scan through most tools drops the label properties, or
  welds vertices and renumbers the faces the labels were written for. Either way the
  labels no longer describe the mesh. The pipeline checks both and says so.
- **The scan is read roughly y-up**, rolled onto the world's z and then stood on its
  floor: the plane of the faces labelled `floor` is turned flat
  (`WarsawScene.world_T_source`). A reconstruction has no sense of gravity, so the roll
  alone left kitchenlab 6 degrees and kitchen2 56 degrees off vertical. A scan labelling
  no floor is only rolled, and a scan written z-up arrives lying on its side. The turn a
  run used is recorded in its `split.json` as `world_T_source`.

## What else a run needs

| Variable | What for |
|---|---|
| `OPENROUTER_API_KEY` | Questions to every model but the one on denver go to OpenRouter. A run is about a hundred calls; on the default model that is a few cents. |
| `QWEN_API_KEY` | Questions to `Model.QWEN3_8_27B_DENVER`, the group's own server; see [the model](#the-model). |
| `SEMANTIC_DIGITAL_TWIN_DATABASE_URI` | A `postgresql+psycopg://` URI. Each run gets a schema of its own under it, so no run inherits another's tables. |

Without a database, set `PipelineSettings.persist = False`. The run then stops after the
split's report, since every step past it reads a world back.

Somewhere to draw is the third requirement. By default every render opens a window on
the display you are sitting at; the next section is for a machine that has none.

A run rewrites two files of the checkout it runs from: `semantic_digital_twin`'s generated
classes and its ORM interface. Run one pipeline at a time, and not while a test session
is running in the same checkout — each would rewrite what the other just read.

## Running a version of the pipeline

The command above runs the defaults. Any other version is a short script that builds the
settings and runs them, which is also how several versions are compared on one scene:

```python
import logging
from pathlib import Path

from experiments.warsaw.pipeline.pipeline import WarsawPipeline
from experiments.warsaw.pipeline.settings import Model, PipelineSettings

logging.basicConfig(level=logging.INFO, format="%(message)s")
WarsawPipeline(
    settings=PipelineSettings(
        scene_directory=Path("<scene directory>"),
        model=Model.QWEN3_8_27B_DENVER,
        show_the_pictures=False,
        show_the_contested_faces=False,
        headless=True,
    )
).carry_out()
```

Run versions one after another, never side by side: every run rewrites the same generated
classes in the checkout (see above). Settings a script leaves out keep their defaults, so
two runs differ in exactly what their scripts set differently; compare their
`provenance.json` to be sure.

### The model

`PipelineSettings.model` takes a member of `Model`, and each member knows where it is
asked:

| Model | Where | Needs |
|---|---|---|
| `Model.QWEN3_VL_30B` (the default), and every other member | OpenRouter | `OPENROUTER_API_KEY` |
| `Model.QWEN3_8_27B_DENVER` | the group's server on denver, at `localhost:8443` | `QWEN_API_KEY`, and an SSH tunnel open while the run lasts |

The tunnel to denver:

```bash
ssh -N -L 8443:127.0.0.1:8443 <you>@<denver's address>
```

The server answers one question at a time, slower than OpenRouter: a run on the
Articulate3D test scene took 11 to 13 minutes there against 6 to 7 for the 30B on
OpenRouter. It reasons before answering, and its reasoning is allowed on top of
`maximum_answer_tokens`. Every model is asked at temperature zero, with
`presence_penalty` keeping it from repeating itself until the answer is cut off.

### The settings that make a version

| Setting | Default | What it changes |
|---|---|---|
| `show_the_pictures` | `True` | Off, nothing is rendered and the label and classification questions are asked from the label and the measurements alone. |
| `show_the_contested_faces` | `True` | Off, the adjudication questions -- who owns the faces two labels claim, and which of several wholes a part belongs to -- are asked without pictures. Turn it off together with `show_the_pictures` for a run without any pictures. |
| `defer_new_classes` | `True` | On, a label is first read only as a class the ontology already holds, or as none; a new class is proposed afterwards, once per label left without one, with how every other label was read in view, and naming a body never proposes one. Off, a new class may be proposed the moment a label or a body is asked about. |
| `settle_classes_by_label` | `True` | On, a body whose label was read as one class takes that class without a question, and one whose label was left open between several classes (the label's `alternatives`) is asked to choose among them or say none fits. Off, every body is asked what it is. |
| `largest_gap_to_a_whole` | `0.10` | How far, in metres, a whole may be from a part that touches none it could belong to, and still be offered as its whole. |
| `ignore_amendments` | `False` | On, a run starts although the ontology's files carry uncommitted changes, for a run meant to use them. |
| `headless` | `False` | See [below](#when-the-renders-come-back-blank). |

The remaining fields of `PipelineSettings` are documented where they are declared.

### Without pictures

A run with `show_the_pictures` and `show_the_contested_faces` off renders nothing. It is
not much faster with a model that spends its time answering: on denver the label questions took as long without the
pictures as with them, and the run was shorter by the rendering, about two and a half
minutes of a scene.

## When the renders come back blank

A run showing its pictures has trimesh draw each into a window and read the picture back
out of that window's colour buffer. By default
(`PipelineSettings.headless = False`) the window is shown, which works wherever there is
a display to show it on. It fills the screen with renders while the run lasts.

`PipelineSettings.headless = True` keeps the window hidden instead. Whether a hidden
window has a colour buffer worth reading is up to the graphics stack: nothing obliges it
to draw into a window that was never shown, and where it does not, the read comes back
blank instead of failing — trimesh warns of exactly this in `render_scene`. On some
machines a hidden window works and is faster; on others it draws nothing at all.

A blank render would have the run measure every object as invisible and still pay for a
hundred questions about black pictures, so the first one raises `BlankRenderError` and
stops the run before anything is asked. Two ways out of it:

- **`PipelineSettings.headless = False`**, if it was set to `True`, on a machine that has
  a display.
- **`xvfb-run -a python -m experiments.warsaw.pipeline.pipeline`** on a machine that has
  none, including over ssh. It gives the run a display of its own and needs the `xvfb`
  package installed.

## What a run leaves behind

A directory under `pipeline_runs/`, named for when the run started, holding every
question as it was put, every answer as it came back, the renders that went with them,
the bodies' meshes, and at the end:

- `report.md` — what was measured, what was asked, what was built, and what it cost.
- `inspect_world.py` — opens the world the run wrote. Set `graph_page = True` in it to
  also write `world_graph.html` beside it and open it in a browser.

The graph page draws any world, not only a run's: bodies and regions joined by their
connections, and each annotation joined to what its fields hold, zoomable and searchable
by name or class.

```python
from experiments.warsaw.world_graph import WorldGraph

WorldGraph.from_world(world).open_page(Path("world_graph.html"))  # or write_page(...)
```

The page loads its drawing library from a CDN, so viewing it needs a network connection.
Runs from before the setting existed can add the same line to their own `inspect_world.py`.

Plus a database schema named for the same run. `RunSchema.drop` throws it away, beside
deleting the directory.

## The steps, in order

1. **Prepare** — put the ontology back as it is committed, rebuild the ORM, give the run
   a schema.
2. **Measure** — how the labelled objects meet, and one render per label.
3. **Vocabulary** — which class of the ontology each label means, and, with
   `defer_new_classes`, a new class for each label no class was found for.
4. **Measure again**, now knowing the classes: what the ontology settles, what is left
   open.
5. **Adjudicate** — whose the contested faces are, and which whole each part belongs to.
6. **Split** — one body per object, written to the database.
7. **Classify** and **mount** — what each body is, and the hierarchy.

Steps 3, 5 and 7 are the ones that ask the model, and so the ones that cost money on
OpenRouter. Every reply is kept beside the question it answered, and every attempt at a
question under `model_calls/`.
