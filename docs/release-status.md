# PhotoMaster release readiness

This checklist records completed preparation and remaining public-release work. One-image and five-image pipelines completed on a single H20; the [GPU execution report](gpu-validation.md) records the evidence, tested source and limitations. This does not establish that the repository is public or that paper results have been reproduced.

## Prepared in this branch

- [x] Paper-aligned project overview with teaser, architecture, resource links and citation.
- [x] Quick-start instructions distinguishing base models from PhotoMaster fine-tuned components.
- [x] Hosted checkpoint filenames and revisions verified; quick-start download commands map `vlm/` and `i2i/` into the launchers' expected local layout.
- [x] Explicit full-transformer loading mode with strict key/shape validation. All 1,933 BF16 tensors match the Diffusers 0.37.0 schema, including the 2511 `zero_cond_t` configuration, and loaded successfully on H20.
- [x] One-image and five-image single-GPU runs completed with zero renderer failures. The five-image run produced five VLM records, five renderer inputs and five readable images; all five outputs are now shown with qualitative inspection notes and visible failures.
- [x] Observed dependency versions, end-to-end timings and sampled whole-card memory usage documented. All 51 archive payload hashes and 59 offline consistency checks passed; selected evidence, direct dependency constraints, exact wheel hashes and a scoped installation recipe are published. A complete portable lock remains pending.
- [x] Data and PDT guides that document the actual five-image example set and output schema.
- [x] Hosted dataset metadata checked against the release revision: 111 shards, 186,405 rows, one exposed Hugging Face split and eight top-level columns. Row-to-triplet mapping remains unverified.
- [x] Single-photo preparation and CPU data validation.
- [x] Preflight checks for required model imports, FlashAttention/native extension, CUDA and local configuration paths; the whole pipeline checks both stages before generation.
- [x] Both launchers default to one GPU; the renderer selects the published complete checkpoint, with explicit base-renderer opt-in.
- [x] Whole-pipeline runs reject existing directories before execution, including a 1-to-5 sample rerun.
- [x] VLM parsing rejects non-object JSON and non-string/blank prompts while retaining failed records.
- [x] Empty renderer ranks participate in common collectives; completion checks expected file paths. These changes have CPU mock coverage only.
- [x] Conversion failure checks to stop invalid VLM output before image generation.
- [x] 60 CPU regression tests and a CI workflow; these do not exercise model inference.

## Required before a public v1 release

| Priority | Gap observed in the checkout | Completion criterion |
|---|---|---|
| P0 | Root code license absent | Authors select and add the project license; accurately document separate data and weight terms. |
| P1 | Source proof covers recorded hashes/patch, not a complete remote snapshot | The two-file patch and three hashes match `e7c2e15`; a future run should calculate a full source manifest at runtime, including prompts/configs. See the published evidence scope. |
| P1 | Complete portable lock and independent installation-recipe replay pending | The observed image inventory, direct constraints and exact wheel hashes are published; independently validate the consolidated recipe before describing it as a clean-machine guarantee. |
| P0 | Full benchmark labels/evaluator absent | Release the 377-case evaluation assets and scripts, or clearly state that v1 is inference-only. |
| P0 | Training assets and paths incomplete | Provide VLM dataset_info.json plus annotations; convert hosted data into both training formats and validate sample counts. |
| P0 | DiT experiment references nonexistent config/data/photomaster_data.yaml | Publish the intended configuration and verify its dataset mapping rather than silently substitute a different experiment. |
| P0 | Training launchers retain machine assumptions | Remove or parameterize forced bond0 network interfaces; initialize Conda consistently; resolve relative DiT config/output paths before changing directories. |
| P0 | Vendored provenance incomplete | Record exact origins/revisions and retain the required third-party notices. |
| P1 | Decision Inversion count differs across paper passages | Reconcile 43,780 in Table 3 versus 42,000 in Section 4.3 with the actual release. |
| P1 | Hosted row count differs from the paper's triplet count | Explain how the 186,405 hosted rows relate to the 236,904 paper triplets; document filtering, release scope or row-to-triplet mapping as applicable. |
| P1 | Full PDT schema and distributed execution remain unvalidated | Top-level JSON and renderer prompt types are now checked. Validate the remaining PDT field schema and real multi-GPU behavior separately; CPU mocks and single-GPU logs do not establish these. |
| P2 | No lightweight interactive demo or project page | Add a demo after the command-line path is validated; report resource requirements and distinguish generated references from actual reshoots. |

## Additional documentation discrepancies

The hosted [model](https://huggingface.co/WilsonCahu/PhotoMaster/tree/ba78362f91a8c6856a1a4605a9fef04e78c06e63) and [dataset](https://huggingface.co/datasets/qwmqza/PhotoMaster/tree/94646f5fddcc5b9ba5e3c2c9c4d5cb0a84c0fdbd) inventories were accessible on October 13, 2026. They contain the two PhotoMaster components and 111 Parquet shards, respectively, but neither revision includes a resource card or license file. The [official dataset metadata](https://datasets-server.huggingface.co/info?dataset=qwmqza%2FPhotoMaster) reports 186,405 rows in `default/train` and eight top-level columns at the same revision; see [the data guide](data.md). The row-level `split` values, internal `frame_json` structure and mapping to the trainers' input formats still need validation.

The VLM training launcher defaults to `data/`, while parts of its guide describe `training_data/`; neither registry is included. The DiT configuration uses 50 checkpoint steps while its guide describes 1,000 as the launcher default. Those can be different settings, but the released reproduction command must identify which experiment is being reproduced.

The experiment file now identifies the recorded H20 run settings. The evidence covers `autodl_h20_smoke_01` and `autodl_h20_demo_05`; the later workflow fixes have CPU validation only, as described in the execution report.

## Release acceptance run

1. Start from a clean clone on the intended Linux/CUDA machine.
2. Download the exact documented model and data revisions, with no private filesystem paths.
3. Run preflight and the one-image pipeline, inspect the JSON and rendered image, and save the environment/version report.
4. Run the five included examples and verify that all expected outputs exist without VLM or parsing failures.
5. If claiming evaluation reproduction, run the full released benchmark and its scoring scripts separately.
6. Finalize the resource cards, license scope and release notes, then publish the reviewed revision.

A README, CPU tests, or successful model imports alone do not establish reproducibility.
