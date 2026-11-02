# Evaluating the Warsaw pipeline on Articulate3D

Articulate3D annotates the parts of articulated objects (cabinets, doors, drawers,
handles, ...) on the meshes of ScanNet++ scenes, together with the part hierarchy. This
package turns that annotation into the pipeline's input and into ground truth, and scores
finished runs against it.

The ground truth is built from Articulate3D's annotation and from a **protocol** a person
writes once, before any run on the evaluated scenes is looked at. Nothing in it comes from
inference rules or from what a run produced.

## What is fixed by hand, and when

| Fixed before the evaluated runs | Never allowed |
| --- | --- |
| The ontology (a pinned commit; extensions decided on train scenes only) | Changing anything below after looking at results on the evaluated scenes |
| The protocol: what every part name means, parts classified on sight, field choices | Decisions about a single scene that are not written into the protocol |
| Prompts and pipeline settings, tuned on train scenes | Anything derived from a run's output |

Freeze all of it with one git tag before running the evaluated scenes.

## Workflow

1. **List the part names** the annotators used and start a protocol from them. Every
   name is added as `undecided`:

   ```bash
   python -m experiments.warsaw.articulate3d.labels --dataset <articulate3d> \
       --protocol protocol.json
   ```

2. **Rule on every name** in `protocol.json`:
   - `class`: the name means one class of the ontology (`semantic_class`);
   - `ambiguous`: the name covers different kinds of things, so these parts are judged as
     objects and as ends of relations, but not on their class;
   - `ignored`: the evaluation leaves these parts out.

   Optionally, classify parts on sight in `part_classes`; this overrides what the part's
   name means. Two people should classify without seeing the part names or any run.
   Where a whole could hold a part through several fields, add a `field_choices` entry.
   Running the command again adds new names and keeps every ruling already made.

3. **Build the ground truth** of every scene. This fails while a name is undecided, while
   the protocol names a class the ontology lacks, or while a field choice is missing:

   ```bash
   python -m experiments.warsaw.articulate3d.ground_truth --dataset <articulate3d> \
       --protocol protocol.json --output ground_truth/
   ```

4. **Convert each scene** into the pipeline's input. The part hierarchy is stripped from
   the labels, and instance numbers are random:

   ```bash
   python -m experiments.warsaw.articulate3d.convert --dataset <articulate3d> \
       --scannetpp <scannet++ data> --scene <scene id> --output <scene directory> \
       [--disclosure words|neutral] [--overlap disjoint|subtree] [--seed N]
   ```

   - `--disclosure neutral` writes labels such as `label_03`, so the model learns nothing
     from the annotators' words. This tests what the pictures alone tell it.
   - `--overlap subtree` makes each part's label cover the parts below it as well. This
     recreates the overlapping labels a real scan produces, so the ownership step runs.

   ScanNet++ meshes must be the v1 meshes the annotations were made on.

5. **Run the pipeline** on each converted scene. Point
   `PipelineSettings.scene_directory` at it, and run each scene several times.

6. **Score the runs**:

   ```bash
   python -m experiments.warsaw.articulate3d.score <run> [<run> ...] \
       --ground-truth ground_truth/ --summary summary.json
   ```

## What the score means

Bodies are paired with annotated parts one to one, by the faces they share (at least half
of the faces either covers, by default). Class and position play no part in the pairing,
so a misclassified body is still found and counted as a misclassification.

| Number | Meaning |
| --- | --- |
| parts found | parts paired with a body |
| fragmented parts / merged bodies | parts split over several bodies / bodies covering several parts |
| bodies outside | bodies lying on faces the annotation does not cover; not judged |
| class accuracy | found parts given exactly their class |
| classes recognised | the same, out of every part whose class is judged, found or not |
| hierarchical precision / recall | partial credit for an answer on the right branch of the class hierarchy |
| answers standing for no class | generated classes with no single class of the ontology as their base |
| relation precision | the run's part-whole mounts that are relations of the part hierarchy |
| relation recall | relations of the part hierarchy the run made |
| expressible | relations the ontology can hold at all: the most any run can make |
| through the same field | correct mounts made through the field the ontology holds them in |

A class the run generated is judged as the one class of the ontology it derives from,
with mixins left out. A share with nothing to divide by is reported as `n/a`, never as a
perfect score.

## Not covered yet

- **Joints.** The pipeline mounts every part with a fixed connection, so the annotated
  axes, origins and ranges are not scored.
- **Held-out ontology classes.** Testing class generation by hiding classes from the
  taxonomy the pipeline sees needs a change to the taxonomy export.
- **Confidence intervals.** Spread over scenes and over repeated runs is not computed yet.
