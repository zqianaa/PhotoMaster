# PhotoMaster data and benchmark

## Resources and release scope

The dataset resource is [qwmqza/PhotoMaster](https://huggingface.co/datasets/qwmqza/PhotoMaster). Its [file inventory at revision `94646f5fddcc5b9ba5e3c2c9c4d5cb0a84c0fdbd`](https://huggingface.co/datasets/qwmqza/PhotoMaster/tree/94646f5fddcc5b9ba5e3c2c9c4d5cb0a84c0fdbd) was verified on October 13, 2026: it is publicly accessible without an access request and contains 111 Parquet files, named `photomaster-00001-of-00111.parquet` through `photomaster-00111-of-00111.parquet`. No dataset card or license file is included at that revision.

The official Hugging Face Dataset Viewer reports **186,405 rows** in the `default` configuration's `train` split, with 85,047,026,604 bytes of original Parquet files (about 85.05 GB). Its [info response](https://datasets-server.huggingface.co/info?dataset=qwmqza%2FPhotoMaster) has an `x-revision` header matching the revision above; the [split inventory](https://datasets-server.huggingface.co/splits?dataset=qwmqza%2FPhotoMaster) and [size report](https://datasets-server.huggingface.co/size?dataset=qwmqza%2FPhotoMaster) agree on the single exposed split and row count. These are service-reported metadata, not a local scan of all shards.

The reported top-level schema is:

| Column | Type |
|---|---|
| `sample_id` | string |
| `split` | string |
| `unsplash_photo_id` | string |
| `unsplash_url` | string |
| `series_idx` | string |
| `label_idx` | string |
| `frame_webp` | binary |
| `frame_json` | string |

The row-level `split` column is distinct from the Hugging Face `train` split. Its values/distribution, the internal structure of `frame_json`, and the mapping from rows to paired images and PDT supervision have not yet been validated. A training-preparation recipe must establish that mapping before claiming reproduction.

The counts below describe the paper's data engine. The paper's **236,904 triplets** and the upload's **186,405 rows** are different reported totals; their relationship needs an explicit explanation before treating one uploaded row as one paper triplet.

| Category | Decision Inversion | Forward Decision Transfer | Total |
|---|---:|---:|---:|
| Portrait | 9,953 | 169,949 | 179,902 |
| Landscape | 30,600 | 13,905 | 44,505 |
| Still life and others | 3,227 | 9,270 | 12,497 |
| Total | 43,780 | 193,124 | 236,904 |

Source: paper Table 3. Section 4.3 separately states 42,000 pairs for Decision Inversion; the release needs an explanation of this difference or a corrected count.

Each conceptual training triplet contains a novice-side image, PDT supervision, and a professional or professional-style reference. One endpoint may be synthesized. The supervision encodes active dimensions, diagnoses and executable actions; a reference-rendering prompt is also stored.

## Included inference examples

`bench/photomaster_bench/metadata.json` contains **five** records with five corresponding images in `img/`. Their diagnosis/ground-truth fields are null. They support inference smoke runs, not evaluation of the paper's macro F1.

The minimal input format consumed by the VLM worker is:

```json
[
  {
    "index": 1,
    "original_img_path": "img/input.png",
    "save_name": "input.png"
  }
]
```

Image paths resolve against the benchmark directory. Indices and output filenames should be unique. `prepare_input.py` creates this layout for a personal image; `check_setup.py --data-only --data-dir ...` validates metadata and path existence.

## Full PhotoMaster-Bench

The paper reports 377 human-curated cases covering the 12 PDT dimensions. To reproduce the diagnosis results, a release needs the benchmark images or retrieval instructions, expert dimension labels, split/provenance information, and the exact evaluation protocol/scripts. Those evaluation annotations and scripts are not present in this checkout.

Keep the full benchmark separate from training data, publish the overlap/deduplication procedure, and identify any samples excluded from redistribution. Do not label the five examples as the full benchmark.

## Training formats

The VLM trainer consumes a LLaMA-Factory ShareGPT dataset registry through `DATASET_DIR`. That registry and its referenced training annotations are not bundled in this checkout. The DiT trainer reads image archives and parquet annotations through the `datset_configs` schema in `config/data/*.yaml` (the spelling is required by the current loader).

The current YAML files contain cluster-specific paths. A portable release needs a conversion/validation script from the actual hosted files into both training layouts, with a small example for each. Replacing path prefixes alone is not sufficient unless the hosted schema matches.

A dataset card should state the released count and splits, source provenance, synthetic-image construction, filtering, label schema, permitted uses, limitations, and the applicable data terms. Record a dataset revision and checksums for the released artifacts.
