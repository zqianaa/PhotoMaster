# H20 execution evidence — October 13, 2026

The original 9,265,781-byte archive was retrieved on October 14. All 51 payload files match the size and SHA-256 entries in [its manifest](archive-manifest.json), with zero missing files. The archive itself has SHA-256 `06e4798d6aea219146fa1f1b4ac0e89765f6946ec894a90525eec30183735ade`. Internal hashes verify archive consistency; they are not a signature from an independent source.

- [Offline consistency audit](verification.json): 59 checks passed, with scope and schema caveats.
- [Archive verification](archive-verification.json) and [weight revisions](download-revisions.json).
- [One-image evidence](autodl_h20_smoke_01/) and [five-image evidence](autodl_h20_demo_05/): original pipeline logs, GPU CSVs, settings, VLM JSON, renderer JSONL and recorded source patch/hashes. Generic remote paths are preserved as evidence, not installation instructions.
- [All five generated comparisons](../../examples.md) and [execution report](../../gpu-validation.md).
- [Observed environment and installation recipe](../../../environments/inference-h20-cu128/README.md).

This directory publishes a selected subset of the original archive, plus derived verification reports. Its archive manifest also lists locally retained files that are not republished here. Generated image bytes are under `docs/assets/results/h20-20261013/`; original inputs remain under `bench/photomaster_bench/img/`. The one-image smoke output was decoded and verified locally, but the gallery uses the five-image run.

The recorded three source hashes match `e7c2e15`, and the recorded patch exactly matches the two inference-file changes from `01f7eed` to `e7c2e15`. The helper copied pre-existing provenance files into each run directory; this is evidence of patch/hash correspondence, not a complete independently captured remote source tree. Model tensor files were not included in this archive.

All six VLM records are parseable and supply the exact prompts used by the renderer. Four records nevertheless contain a total of five inactive `Attribution: "None"` string fields where the repository's reference prompt specifies arrays. The converter accepts these records; parseability and rendering success do not establish strict schema compliance or advice quality.
