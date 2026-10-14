"""CPU-only regression tests for renderer defaults and FlashAttention preflight."""

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "launcher_preflight_setup", ROOT / "scripts/check_setup.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class RendererLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repo with spaces"
        script_dir = self.root / "scripts/inference"
        script_dir.mkdir(parents=True)
        self.launcher = script_dir / "run_qwen_edit_infer.sh"
        shutil.copyfile(ROOT / "scripts/inference/run_qwen_edit_infer.sh", self.launcher)
        (self.root / "models/Qwen-Image-Edit-2511").mkdir(parents=True)
        self.checkpoint = self.root / "models/ft_ckpts/i2i/photomaster_i2i.safetensors"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()
        result_dir = self.root / "bench/photomaster_bench/results/vlm"
        result_dir.mkdir(parents=True)
        (result_dir / "dit_input.jsonl").write_text("{}\n")
        self.command_log = self.root / "command.log"
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        for command in ("python", "torchrun"):
            stub = fake_bin / command
            stub.write_text(
                f'#!/bin/sh\nprintf "%s\\n" {command} "$@" > "$TEST_COMMAND_LOG"\n'
            )
            stub.chmod(0o755)
        self.environment = {
            "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "TEST_COMMAND_LOG": str(self.command_log),
        }

    def launch(self, **overrides):
        return subprocess.run(
            ["bash", str(self.launcher)], env=dict(self.environment, **overrides),
            text=True, capture_output=True, check=False,
        )

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.command_log.read_text().splitlines()

    def test_default_uses_one_process_and_complete_published_checkpoint(self):
        command = self.command(self.launch())
        self.assertEqual(command[0], "python")
        self.assertEqual(command[command.index("--ckpt-path") + 1], str(self.checkpoint))
        self.assertIn("--full-transformer-checkpoint", command)

    def test_missing_published_checkpoint_fails_without_running_base_renderer(self):
        self.checkpoint.unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FT checkpoint not found", result.stdout)
        self.assertFalse(self.command_log.exists())

    def test_base_renderer_requires_deliberate_opt_in(self):
        self.checkpoint.unlink()
        command = self.command(self.launch(ALLOW_BASE_DIT="1"))
        self.assertEqual(command[0], "python")
        self.assertNotIn("--ckpt-path", command)
        self.assertNotIn("--full-transformer-checkpoint", command)

    def test_explicit_checkpoint_preserves_partial_mode_by_default(self):
        partial = self.checkpoint.with_name("partial.safetensors")
        partial.touch()
        command = self.command(self.launch(FT_CKPT_PATH=str(partial)))
        self.assertEqual(command[command.index("--ckpt-path") + 1], str(partial))
        self.assertNotIn("--full-transformer-checkpoint", command)

    def test_explicit_checkpoint_can_select_full_mode(self):
        command = self.command(self.launch(FT_CKPT_PATH=str(self.checkpoint), FT_CKPT_IS_FULL="1"))
        self.assertIn("--full-transformer-checkpoint", command)

    def test_explicit_partial_mode_overrides_default_checkpoint_mode(self):
        command = self.command(self.launch(FT_CKPT_IS_FULL="0"))
        self.assertIn("--ckpt-path", command)
        self.assertNotIn("--full-transformer-checkpoint", command)

    def test_multi_gpu_is_still_available_explicitly(self):
        command = self.command(self.launch(GPU_MODE="multi_gpu", GPUS_PER_NODE="2"))
        self.assertEqual(command[0], "torchrun")
        self.assertIn("--nproc_per_node=2", command)

    def test_multi_node_uses_explicit_rendezvous_port(self):
        command = self.command(self.launch(
            GPU_MODE="multi_node", GPUS_PER_NODE="2", NUM_NODES="3",
            MASTER_ADDR="test-host", MASTER_PORT="29601",
        ))
        self.assertEqual(command[0], "torchrun")
        self.assertIn("--nnodes=3", command)
        self.assertIn("--nproc_per_node=2", command)
        self.assertIn("--rdzv_endpoint=test-host:29601", command)

    def test_invalid_or_conflicting_base_opt_in_fails_before_worker(self):
        for overrides in (
            {"ALLOW_BASE_DIT": "yes"},
            {"ALLOW_BASE_DIT": "1", "FT_CKPT_PATH": str(self.checkpoint)},
            {"ALLOW_BASE_DIT": "1", "FT_CKPT_IS_FULL": "1"},
        ):
            with self.subTest(overrides=overrides):
                result = self.launch(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.command_log.exists())


class FlashAttentionPreflightTests(unittest.TestCase):
    def test_imports_cuda_extension_as_well_as_python_package(self):
        with mock.patch.object(setup.importlib.metadata, "version", return_value="2.8.3"), \
                mock.patch.object(setup.importlib, "import_module") as importer, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.check_flash_attention(), [])
        self.assertEqual(importer.call_args_list, [mock.call("flash_attn"), mock.call("flash_attn_2_cuda")])

    def test_missing_package_is_reported_with_install_guidance(self):
        with mock.patch.object(
            setup.importlib.metadata, "version",
            side_effect=setup.importlib.metadata.PackageNotFoundError("flash-attn"),
        ):
            errors = setup.check_flash_attention()
        self.assertEqual(len(errors), 1)
        self.assertIn("flash_attention_2", errors[0])
        self.assertIn("matching", errors[0])

    def test_binary_import_failure_is_reported(self):
        with mock.patch.object(setup.importlib.metadata, "version", return_value="2.8.3"), \
                mock.patch.object(setup.importlib, "import_module", side_effect=[object(), ImportError("undefined symbol")]), \
                contextlib.redirect_stdout(io.StringIO()):
            errors = setup.check_flash_attention()
        self.assertEqual(len(errors), 1)
        self.assertIn("undefined symbol", errors[0])
        self.assertIn("C++ ABI", errors[0])

    def test_data_only_does_not_require_flash_attention(self):
        with mock.patch.object(sys, "argv", ["check_setup.py", "--data-only"]), \
                mock.patch.object(setup, "check_flash_attention") as check, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main(), 0)
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
