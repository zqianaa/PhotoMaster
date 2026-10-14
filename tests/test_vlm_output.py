"""CPU-only checks for isolating invalid responses before renderer conversion."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse = load_module("vlm_output", "scripts/inference/vlm_output.py").parse_vlm_output
convert = load_module("vlm_convert", "scripts/inference/vlm_to_dit_bench_reformat.py").convert


class VLMOutputTests(unittest.TestCase):
    def setUp(self):
        self.response = {
            "Issues_Analysis": [
                {"Dimension": "A1 Subject Placement", "Detail": "None"},
                {"Dimension": "C1 Shot Size", "Detail": "The subject is too small."},
            ],
            "Professional_Suggestions": [
                {"Dimension": "C1 Shot Size", "Detail": "Move closer."},
            ],
            "Image_Editing_Prompt": " Preserve the subject and use a medium shot. ",
        }

    def assert_rejected(self, raw):
        result = parse(raw)
        self.assertTrue(result["parse_error"])
        self.assertEqual(result["model_output_raw"], raw)
        for field in ("Issues_Analysis", "Professional_Suggestions", "Image_Editing_Prompt"):
            self.assertEqual(result[field], "")
        return result

    def test_valid_response_preserves_prompt_and_display_format(self):
        raw = json.dumps(self.response)
        result = parse(raw)
        self.assertNotIn("parse_error", result)
        self.assertNotIn("error", result)
        self.assertEqual(result["model_output_raw"], raw)
        self.assertEqual(result["Issues_Analysis"], "【C1 Shot Size】The subject is too small.")
        self.assertEqual(result["Professional_Suggestions"], "【C1 Shot Size】Move closer.")
        self.assertEqual(result["Image_Editing_Prompt"], self.response["Image_Editing_Prompt"])

    def test_non_object_json_is_rejected_without_raising(self):
        for value in (None, [], [self.response], "text", 1, True):
            with self.subTest(value=value):
                self.assert_rejected(json.dumps(value))

    def test_missing_non_string_and_blank_prompts_are_rejected(self):
        missing = dict(self.response)
        del missing["Image_Editing_Prompt"]
        self.assert_rejected(json.dumps(missing))
        for prompt in (None, [], ["a prompt"], {"text": "a prompt"}, 1, False, "", " \t\n"):
            with self.subTest(prompt=prompt):
                self.assert_rejected(json.dumps(dict(self.response, Image_Editing_Prompt=prompt)))

    def test_malformed_json_and_incomplete_fences_are_rejected(self):
        for raw in ("", "not JSON", '{"Image_Editing_Prompt":',
                    "```json\n" + json.dumps(self.response),
                    json.dumps(self.response) + "\nextra text"):
            with self.subTest(raw=raw):
                self.assert_rejected(raw)

    def test_fenced_json_preserves_backticks_inside_prompt(self):
        response = dict(self.response, Image_Editing_Prompt="Keep the literal ```json text on the sign.")
        for opening in ("```json", "```", "```JSON"):
            raw = " \n" + opening + "\n" + json.dumps(response) + "\n```\n "
            with self.subTest(opening=opening):
                result = parse(raw)
                self.assertNotIn("parse_error", result)
                self.assertEqual(result["Image_Editing_Prompt"], response["Image_Editing_Prompt"])
                self.assertEqual(result["model_output_raw"], raw)

    def test_existing_non_list_display_fields_remain_compatible(self):
        response = dict(self.response, Issues_Analysis="Existing diagnosis text.",
                        Professional_Suggestions={"note": "Existing advice."})
        result = parse(json.dumps(response))
        self.assertEqual(result["Issues_Analysis"], response["Issues_Analysis"])
        self.assertEqual(result["Professional_Suggestions"], str(response["Professional_Suggestions"]))
        self.assertNotIn("parse_error", result)

    def test_invalid_detail_type_cannot_interrupt_batch(self):
        for detail in (123, ["advice"], {"text": "advice"}):
            response = dict(self.response, Issues_Analysis=[{"Dimension": "C1 Shot Size", "Detail": detail}])
            with self.subTest(detail=detail):
                self.assert_rejected(json.dumps(response))

    def test_generation_error_preserves_original_error(self):
        raw = "ERROR: generation failed"
        result = parse(raw)
        self.assertEqual(result["error"], raw)
        self.assertEqual(result["model_output_raw"], raw)
        self.assertEqual(result["Image_Editing_Prompt"], "")
        self.assertNotIn("parse_error", result)

    def test_bad_sample_is_retained_but_converter_rejects_entire_input(self):
        # The same per-sample function used by the worker continues past null
        # and preserves later valid records for diagnosis/retry.
        responses = [json.dumps(self.response), "null", json.dumps(self.response)]
        records = [dict(parse(raw), index=index, original_img_path=f"img/{index}.png")
                   for index, raw in enumerate(responses, 1)]
        self.assertEqual(len(records), 3)
        self.assertIn("parse_error", records[1])
        self.assertNotIn("parse_error", records[2])
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "vlm.json", Path(temp) / "dit.jsonl"
            source.write_text(json.dumps({"results": records}))
            output.write_text("previous valid output")
            with self.assertRaisesRegex(ValueError, "Sample 2"):
                convert(source, output)
            self.assertEqual(output.read_text(), "previous valid output")


if __name__ == "__main__":
    unittest.main()
