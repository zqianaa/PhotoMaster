"""CPU-only checks for the opt-in complete transformer loading safeguards."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeTensor:
    def __init__(self, is_meta=False):
        self.is_meta = is_meta


# These tests exercise metadata/argument validation, requiring no torch install
# and allocating no model tensors. Actual loading needs the GPU smoke environment.
torch_stub = types.ModuleType("torch")
torch_stub.is_tensor = lambda value: isinstance(value, FakeTensor)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen_edit_full_checkpoint_checks", ROOT / "scripts/inference/qwen_edit_infer.py"
)
worker = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"torch": torch_stub}):
    spec.loader.exec_module(worker)


class FullTransformerSchemaTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"img_in.weight": (3, 4), "img_in.bias": (3,)}

    def test_complete_header_matches_list_or_tuple_shapes(self):
        worker._validate_full_transformer_shapes(
            {"img_in.weight": [3, 4], "img_in.bias": [3]}, self.expected
        )

    def test_partial_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing=1"):
            worker._validate_full_transformer_shapes({"img_in.bias": [3]}, self.expected)

    def test_equal_key_count_does_not_hide_wrong_names(self):
        with self.assertRaisesRegex(ValueError, "unexpected=1"):
            worker._validate_full_transformer_shapes(
                {"img_in.bias": [3], "another.weight": [3, 4]}, self.expected
            )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape_mismatches=1"):
            worker._validate_full_transformer_shapes(
                {"img_in.bias": [3], "img_in.weight": [4, 3]}, self.expected
            )

    def test_uniform_training_prefixes_can_be_normalized(self):
        shapes = {f"module.pipe.dit.{key}": value for key, value in self.expected.items()}
        for prefix in ("module.", "pipe.dit.", "dit.", "transformer."):
            shapes = worker._strip_state_dict_prefix(shapes, prefix)
        worker._validate_full_transformer_shapes(shapes, self.expected)

    def test_mixed_prefixes_are_not_silently_accepted(self):
        shapes = {"img_in.bias": [3], "transformer.img_in.weight": [3, 4]}
        shapes = worker._strip_state_dict_prefix(shapes, "transformer.")
        with self.assertRaises(ValueError):
            worker._validate_full_transformer_shapes(shapes, self.expected)


class MetaResidueTests(unittest.TestCase):
    def model(self, meta_kind=None):
        model = types.SimpleNamespace()
        model.named_parameters = lambda: [("weight", FakeTensor(meta_kind == "parameter"))]
        model.named_buffers = lambda: [("buffer", FakeTensor(meta_kind == "buffer"))]
        rope = types.SimpleNamespace(pos_freqs=FakeTensor(meta_kind == "unregistered_rope"))
        model.named_modules = lambda: [("", model), ("pos_embed", rope)]
        return model

    def test_cpu_tensors_are_accepted(self):
        worker._assert_no_meta_tensors(self.model())

    def test_parameters_buffers_and_unregistered_rope_are_checked(self):
        for kind in ("parameter", "buffer", "unregistered_rope"):
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, "meta tensors"):
                worker._assert_no_meta_tensors(self.model(kind))


class FullTransformerArgumentTests(unittest.TestCase):
    ARGS = ["qwen_edit_infer.py", "--model-path", "models/base", "--jsonl-path", "input.jsonl", "--output-dir", "out"]

    def test_partial_mode_remains_default(self):
        with mock.patch.object(sys, "argv", self.ARGS + ["--ckpt-path", "partial.safetensors"]):
            self.assertFalse(worker.parse_args().full_transformer_checkpoint)

    def test_full_mode_requires_safetensors_checkpoint(self):
        for extra in ([], ["--ckpt-path", "checkpoint.pt"]):
            with self.subTest(extra=extra), mock.patch.object(
                sys, "argv", self.ARGS + ["--full-transformer-checkpoint"] + extra
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                worker.parse_args()
            self.assertEqual(error.exception.code, 2)

    def test_full_mode_is_explicit(self):
        with mock.patch.object(
            sys, "argv", self.ARGS + ["--full-transformer-checkpoint", "--ckpt-path", "full.safetensors"]
        ):
            self.assertTrue(worker.parse_args().full_transformer_checkpoint)


if __name__ == "__main__":
    unittest.main()
