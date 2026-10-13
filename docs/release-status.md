# PhotoMaster release readiness

This checklist describes the local release-preparation branch based on commit `5ad5a81`. It does not claim that the remote repository is public or that the GPU pipeline has been validated here.

## Prepared in this branch

- [x] Paper-aligned project overview with teaser, architecture, resource links and citation.
- [x] Quick-start instructions distinguishing base models from PhotoMaster fine-tuned components.
- [x] Data and PDT guides that document the actual five-image example set and output schema.
- [x] Single-photo preparation and CPU data validation.
- [x] Preflight checks for required model imports, CUDA and local configuration paths.
- [x] Conversion failure checks to stop invalid VLM output before image generation.
- [x] Lightweight CPU tests and a CI workflow; these do not exercise model inference.

## Required before a public v1 release

| Priority | Gap observed in the checkout | Completion criterion |
|---|---|---|
| P0 | Root code license absent | Authors select and add the project license; accurately document separate data and weight terms. |
| P0 | Hosted checkpoint inventory unverified | Map both PhotoMaster components to exact files, base models, revisions and checksums; provide tested download commands. |
| P0 | No validated inference environment lock | Run one example in a fresh CUDA environment, record versions/GPU/peak memory/runtime, and publish the working lock. |
| P0 | Full benchmark labels/evaluator absent | Release the 377-case evaluation assets and scripts, or clearly state that v1 is inference-only. |
| P0 | Training assets and paths incomplete | Provide VLM dataset_info.json plus annotations; convert hosted data into both training formats and validate sample counts. |
| P0 | DiT experiment references nonexistent config/data/photomaster_data.yaml | Publish the intended configuration and verify its dataset mapping rather than silently substitute a different experiment. |
| P0 | Training launchers retain machine assumptions | Remove or parameterize forced bond0 network interfaces; initialize Conda consistently; resolve relative DiT config/output paths before changing directories. |
| P0 | Vendored provenance incomplete | Record exact origins/revisions and retain the required third-party notices. |
| P1 | Decision Inversion count differs across paper passages | Reconcile 43,780 in Table 3 versus 42,000 in Section 4.3 with the actual release. |
| P1 | No released prediction/reference gallery | Save real outputs from the released checkpoint on 3–5 representative cases, labeled as generated references. |
| P1 | Inference can reuse stale cached results | Use distinct run names now; later store and validate a manifest of checkpoints, prompts and settings before resuming. |
| P2 | No lightweight interactive demo or project page | Add a demo after the command-line path is validated; report resource requirements and distinguish generated references from actual reshoots. |

## Additional documentation discrepancies

The VLM training launcher defaults to `data/`, while parts of its guide describe `training_data/`; neither registry is included. The DiT configuration uses 50 checkpoint steps while its guide describes 1,000 as the launcher default. Those can be different settings, but the released reproduction command must identify which experiment is being reproduced.

The inference environment file describes a tracked reference run, but no reference result is included in this checkout. Its values should be called recorded parameters until the corresponding outputs/checkpoint provenance are supplied.

## Release acceptance run

1. Start from a clean clone on the intended Linux/CUDA machine.
2. Download the exact documented model and data revisions, with no private filesystem paths.
3. Run preflight and the one-image pipeline, inspect the JSON and rendered image, and save the environment/version report.
4. Run the five included examples and verify that all expected outputs exist without VLM or parsing failures.
5. If claiming evaluation reproduction, run the full released benchmark and its scoring scripts separately.
6. Finalize the resource cards, license scope and release notes, then publish the reviewed revision.

A README, CPU tests, or successful model imports alone do not establish reproducibility.
