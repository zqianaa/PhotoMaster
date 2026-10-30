# Third-party components and figure attribution

## Code

| Component | Location | Recorded provenance |
|---|---|---|
| LLaMA-Factory | `scripts/vlm_train/LLaMA-Factory/` | Its training README records upstream commit `1317057` and five local patches; its LICENSE is retained. |
| Qwen image-edit pipeline | `scripts/inference/vendor/qwenimage/pipeline_qwenimage_edit_plus.py` | The file retains the Qwen-Image/Hugging Face copyright and Apache-2.0 header; the exact upstream revision and patch summary still need recording. |
| DiT trainer | `scripts/dit_train/trainer/` | The training guide identifies DiffSynth 2.0.1 as a runtime dependency. The origin/revision and applicable notices for the copied trainer files still need documenting. |

These notices do not define a license for the entire PhotoMaster project. Code, data, base models and derived weights require their own accurate release terms.

## Figures

`docs/assets/teaser.png` and `docs/assets/framework.png` are crops of Figures 1 and 4 from the author-supplied camera-ready paper:

Zian Qian, Cheuk Him Chau, Siyue Xie, Chufeng Xiao, Qifeng Chen, Rui Liu, and Chenyang Lei. *PhotoMaster: Learning Holistic Capture-Time Decisions for Guided Photography Reshooting.* 2026. DOI: [10.1145/3842543](https://doi.org/10.1145/3842543).

The paper states CC BY 4.0. These assets are cropped and rasterized, with figure content unchanged. Figure 1's caption states that its participant photographs are used with permission. Confirm that the intended repository distribution is covered before public release.
