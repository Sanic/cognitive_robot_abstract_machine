# RoboKudo Benchmarks

This folder contains benchmark suites for non-interactive RoboKudo evaluation.
Benchmarks are AnalysisEngine-centric: each suite points to an AnalysisEngine
descriptor and optionally defines controlled injections (e.g. CollectionReader
camera config override).

## Layout

- `suites/<suite_name>/spec.json`:
  Benchmark definition file (validated against the v1 schema).

## Validation

Use the benchmark validator before running any suite:

```bash
python -m robokudo.benchmarking.cli validate /abs/path/to/spec.json
```

The validator checks:

- JSON schema compliance (`benchmark_spec.v1.schema.json`)
- Additional semantic constraints (e.g., duplicate `sample_index`)

## Notes

- Benchmark artifacts should be written to `robokudo/.benchmark_runs/` (gitignored).
- Specs are intentionally strict so AI agents can iteratively fix validation errors.

## AnalysisEngine-Centric Fields

- `analysis_engine.ros_pkg_name` and `analysis_engine.module` identify the
  AnalysisEngine to load via `ModuleLoader.load_ae(...)`.
- `injections.collection_reader` can keep the AE default behavior or enforce a
  specific camera config via `CrDescriptorFactory`.
- `injections.collection_reader.target_annotator_names` can restrict injection
  to selected `CollectionReaderAnnotator` node names when an AE has multiple readers.
