# Photography Decision Taxonomy and output format

The paper groups 12 dimensions into six domains. The released prompt uses A1–F2 labels; the paper uses domain-based notation. This mapping preserves the strings expected by the existing models.

| Code and canonical dimension | Paper notation | Domain |
|---|---|---|
| A1 Subject Placement | S_cont | Subject |
| A2 Pose & Expression | S_tech | Subject |
| B1 Camera Position | V_cont | Viewpoint |
| B2 Shooting Angle | V_tech | Viewpoint |
| C1 Shot Size | FO_cont | Framing and optics |
| C2 Depth of Field | FO_tech | Framing and optics |
| D1 Motion Treatment | T_cont | Time |
| D2 Decisive Moment | T_tech | Time |
| E1 Lighting Utilization | LC_cont | Light and color |
| E2 Color & Tonal Key | LC_tech | Light and color |
| F1 Background | SG_cont | Scene and geometry |
| F2 Compositional Structure | SG_tech | Scene and geometry |

## Model response

The [system prompt](../scripts/inference/prompts/system_prompt.txt) requests:

- `Issues_Analysis`: all 12 canonical dimensions, with `Detail` equal to the literal string `"None"` when acceptable.
- `Professional_Suggestions`: actions for A1–E2. F1 and F2 are diagnosis-only; their issue entries can use `Attribution` to map to actionable dimensions.
- `Image_Editing_Prompt`: an English rendering prompt describing the desired visual reference.

The action instructions are for the photographer; the rendering prompt is for the image generator. They serve different interfaces.

## Saved output

`vlm_output.json` is a run-level object with a `results` array. Each record retains its original input metadata and adds the three output fields, raw model text, timing and diagnostics. **The current worker serializes `Issues_Analysis` and `Professional_Suggestions` as formatted strings**, even when the model returned arrays. Do not assume that their on-disk values are structured arrays; inspect `model_output_raw` if you need the original JSON response.

The conversion step uses `Image_Editing_Prompt` to produce JSONL:

```json
{"query_id":"1","caption":["The rendering prompt"],"image":["img/input.png"],"save_name":"input.png","task":""}
```

This is a format example, not an actual model prediction. Results containing generation errors, parse errors or blank rendering prompts are rejected before conversion. A syntactically valid response still needs review for PDT compliance and physical feasibility.
