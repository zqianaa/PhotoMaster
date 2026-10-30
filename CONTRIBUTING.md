# Contributing to PhotoMaster

Use a focused issue or pull request describing the problem, the expected behavior, and a small reproducible example. For inference failures, include the command, OS, Python/package versions, GPU and memory, model revisions, and the relevant error. Remove credentials, personal paths and private photos from shared logs.

Run the lightweight checks from the repository root:

```bash
python -m pip install Pillow
python -m unittest discover -s tests -v
python scripts/check_setup.py --data-only
```

GPU changes also need an inference run with the relevant checkpoints. State clearly when GPU validation has not been performed. Keep vendored changes focused and document their upstream source and local differences. See [release status](docs/release-status.md) for work needed before the public release.
