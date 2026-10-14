"""CPU mocks for renderer sharding, collective order and output completeness.

These tests exercise main() without torch, model weights or CUDA. They do not
validate NCCL transport or real multi-process/GPU execution.
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen_edit_distributed_checks", ROOT / "scripts/inference/qwen_edit_infer.py"
)
worker = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    spec.loader.exec_module(worker)


class CountTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return types.SimpleNamespace(item=lambda: self.values[index])


class RendererMainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()
        self.jsonl_path = self.root / "dit_input.jsonl"

    def samples(self, count):
        return [
            {"query_id": str(index), "save_name": f"photo_{index}.png",
             "image": ["source.png"], "caption": [f"Edit sample {index}"]}
            for index in range(count)
        ]

    def write_output(self, sample):
        path = Path(worker.get_output_path(sample, str(self.output_dir)))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(path)

    def run_rank(self, samples, rank=0, world_size=1, totals=None,
                 peer_work=None, save_output=True, valid_images=True):
        self.jsonl_path.write_text("".join(json.dumps(s) + "\n" for s in samples))
        events = []
        submitted_counts = []
        output = io.StringIO()
        image = Image.new("RGB", (16, 16)) if save_output else mock.Mock()
        pipeline = mock.Mock(return_value=types.SimpleNamespace(images=[image]))
        pipeline_class = types.SimpleNamespace(
            __module__="mock_pipeline", from_pretrained=mock.Mock(return_value=pipeline)
        )
        pipeline_loader = mock.Mock(return_value=pipeline_class)

        def init_distributed():
            events.append("init")
            return rank, world_size, rank

        def all_reduce(counts, op):
            events.append("all_reduce")
            self.assertEqual(op, "sum")
            submitted_counts.append(list(counts.values))
            # A peer's saves complete before its all_reduce in the real worker.
            if peer_work is not None:
                peer_work()
            if totals is not None:
                counts.values[:] = totals

        torch_mock = types.SimpleNamespace(
            bfloat16="bfloat16", long="long",
            tensor=lambda values, **kwargs: CountTensor(values),
            Generator=lambda **kwargs: types.SimpleNamespace(manual_seed=lambda seed: object()),
            inference_mode=contextlib.nullcontext,
            distributed=types.SimpleNamespace(
                ReduceOp=types.SimpleNamespace(SUM="sum"),
                all_reduce=all_reduce,
                barrier=lambda: events.append("barrier"),
                destroy_process_group=lambda: events.append("destroy"),
            ),
        )
        argv = [
            "qwen_edit_infer.py", "--model-path", "unused-model",
            "--jsonl-path", str(self.jsonl_path), "--output-dir", str(self.output_dir),
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(mock.patch.object(worker, "torch", torch_mock))
            stack.enter_context(mock.patch.object(worker, "init_distributed", init_distributed))
            stack.enter_context(mock.patch.object(worker, "load_qwen_image_edit_plus_pipeline_class", pipeline_loader))
            stack.enter_context(mock.patch.object(worker, "load_images", return_value=[Image.new("RGB", (16, 16))] if valid_images else []))
            stack.enter_context(contextlib.redirect_stdout(output))
            exit_code = 0
            try:
                worker.main()
            except SystemExit as error:
                exit_code = error.code
        return types.SimpleNamespace(
            exit_code=exit_code, output=output.getvalue(), events=events,
            submitted_counts=submitted_counts, pipeline_loader=pipeline_loader,
            pipeline=pipeline,
        )

    def assert_clean_collectives(self, result, counts):
        self.assertEqual(result.events, ["init", "all_reduce", "barrier", "destroy"])
        self.assertEqual(result.submitted_counts, [counts])

    def test_more_ranks_than_samples_use_same_collectives(self):
        samples = self.samples(1)
        for rank in range(3):
            with self.subTest(rank=rank):
                result = self.run_rank(samples, rank=rank, world_size=3, totals=[1, 0])
                self.assertEqual(result.exit_code, 0)
                self.assert_clean_collectives(result, [1, 0] if rank == 0 else [0, 0])
                self.assertEqual(result.pipeline_loader.call_count, 1 if rank == 0 else 0)

    def test_completed_main_shard_waits_for_peer_work_and_checks_all_paths(self):
        samples = self.samples(2)
        self.write_output(samples[0])
        result = self.run_rank(
            samples, world_size=2, totals=[1, 0],
            peer_work=lambda: self.write_output(samples[1]),
        )
        self.assertEqual(result.exit_code, 0)
        self.assert_clean_collectives(result, [0, 0])
        result.pipeline_loader.assert_not_called()
        self.assertIn("Total expected output images present: 2/2", result.output)

    def test_resumed_non_main_shard_only_renders_its_pending_sample(self):
        samples = self.samples(4)
        self.write_output(samples[1])
        result = self.run_rank(samples, rank=1, world_size=2, totals=[1, 0])
        self.assertEqual(result.exit_code, 0)
        self.assert_clean_collectives(result, [1, 0])
        result.pipeline.assert_called_once()
        self.assertEqual(result.pipeline.call_args.kwargs["prompt"], "Edit sample 3")
        self.assertTrue((self.output_dir / "photo_3.png").is_file())
        self.assertFalse((self.output_dir / "photo_0.png").exists())

    def test_all_completed_ranks_skip_loading_but_still_reduce(self):
        samples = self.samples(2)
        for sample in samples:
            self.write_output(sample)
        for rank in range(2):
            with self.subTest(rank=rank):
                result = self.run_rank(samples, rank=rank, world_size=2, totals=[0, 0])
                self.assertEqual(result.exit_code, 0)
                self.assert_clean_collectives(result, [0, 0])
                result.pipeline_loader.assert_not_called()

    def test_empty_input_joins_collectives_without_loading(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                result = self.run_rank([], rank=rank, world_size=2, totals=[0, 0])
                self.assertEqual(result.exit_code, 0)
                self.assert_clean_collectives(result, [0, 0])
                result.pipeline_loader.assert_not_called()

    def test_peer_failure_reaches_idle_rank_and_exits_after_cleanup(self):
        samples = self.samples(1)
        result = self.run_rank(samples, rank=1, world_size=2, totals=[0, 1])
        self.assertEqual(result.exit_code, 1)
        self.assert_clean_collectives(result, [0, 0])
        result.pipeline_loader.assert_not_called()

    def test_local_failure_is_reduced_before_cleanup(self):
        result = self.run_rank(self.samples(1), world_size=2, totals=[0, 1], valid_images=False)
        self.assertEqual(result.exit_code, 1)
        self.assert_clean_collectives(result, [0, 1])
        result.pipeline.assert_not_called()

    def test_unrelated_image_cannot_hide_missing_expected_output(self):
        self.write_output({"save_name": "unrelated.png"})
        result = self.run_rank(self.samples(1), world_size=2, totals=[1, 0], save_output=False)
        self.assertEqual(result.exit_code, 1)
        self.assert_clean_collectives(result, [1, 0])
        self.assertIn("1 expected images missing", result.output)
        self.assertIn("photo_0.png", result.output)

    def test_nested_expected_output_counts_even_when_not_top_level(self):
        samples = [{"query_id": "nested", "save_name": "nested/photo.png"}]
        self.write_output(samples[0])
        result = self.run_rank(samples, world_size=2, totals=[0, 0])
        self.assertEqual(result.exit_code, 0)
        self.assert_clean_collectives(result, [0, 0])
        result.pipeline_loader.assert_not_called()

    def test_single_gpu_completed_run_needs_no_collectives_or_pipeline(self):
        samples = self.samples(1)
        self.write_output(samples[0])
        result = self.run_rank(samples)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.events, ["init"])
        result.pipeline_loader.assert_not_called()
        self.assertIn("Total expected output images present: 1/1", result.output)


if __name__ == "__main__":
    unittest.main()
