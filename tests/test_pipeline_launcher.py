"""Exercise the public pipeline entry point with fake, CPU-only stages."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PipelineLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / 'scripts/inference'
        self.scripts.mkdir(parents=True)
        self.launcher = self.scripts / 'run_pipeline.sh'
        shutil.copyfile(ROOT / 'scripts/inference/run_pipeline.sh', self.launcher)
        self.calls = self.root / 'calls'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        python = self.bin / 'python'
        python.write_text('#!/bin/bash\nprintf "preflight %s\\n" "$*" >> "$CALLS"\nexit "${PREFLIGHT_EXIT:-0}"\n')
        python.chmod(0o755)
        for stage in ('run_vlm_infer.sh', 'run_vlm_to_dit.sh', 'run_qwen_edit_infer.sh'):
            (self.scripts / stage).write_text(
                '#!/bin/bash\nprintf "%s %s %s %s %s\\n" "$(basename "$0")" "$1" "$2" "$RUN_NAME" "$GPU_MODE" >> "$CALLS"\n'
                'exit "${STAGE_EXIT:-0}"\n')
        self.env = {key: value for key, value in os.environ.items() if key in ('PATH', 'HOME', 'TMPDIR')}
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env['PATH'], CALLS=str(self.calls), RUN_NAME='fresh')

    def run_pipeline(self, **env):
        return subprocess.run(['bash', str(self.launcher), 'vlm', 'photomaster_bench'], env={**self.env, **env}, capture_output=True, text=True)

    def test_fresh_pipeline_preflights_both_models_then_uses_one_run_and_single_gpu(self):
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertIn('--vlm-path', calls[0])
        self.assertIn('--dit-path', calls[0])
        self.assertIn('photomaster_i2i.safetensors', calls[0])
        self.assertEqual(calls[1:], ['run_vlm_infer.sh vlm photomaster_bench fresh single',
                                     'run_vlm_to_dit.sh fresh photomaster_bench fresh single',
                                     'run_qwen_edit_infer.sh fresh photomaster_bench fresh single'])

    def test_one_to_five_sample_rerun_rejected_before_any_stage_or_overwrite(self):
        self.assertEqual(self.run_pipeline(NUM_SAMPLES='1').returncode, 0)
        output = self.root / 'bench/photomaster_bench/results/fresh/dit_input.jsonl'
        output.write_text(json.dumps({'caption': ['old prompt']}))
        before = self.calls.read_bytes()
        result = self.run_pipeline(NUM_SAMPLES='5', FORCE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('new RUN_NAME', result.stdout)
        self.assertEqual(self.calls.read_bytes(), before)
        self.assertEqual(json.loads(output.read_text())['caption'], ['old prompt'])

    def test_preflight_failure_prevents_gpu_stages_and_directory_reservation(self):
        result = self.run_pipeline(PREFLIGHT_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertFalse((self.root / 'bench/photomaster_bench/results/fresh').exists())

    def test_stage_failure_stops_downstream_stages(self):
        result = self.run_pipeline(STAGE_EXIT='2')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)

    def test_shared_model_path_rejected_before_any_stage(self):
        result = self.run_pipeline(MODEL_PATH='/one/model')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MODEL_PATH', result.stdout)
        self.assertFalse(self.calls.exists())

    def test_unsafe_run_names_and_unknown_gpu_modes_fail_before_stages(self):
        for env in ({'RUN_NAME': '../escape'}, {'RUN_NAME': 'a/b'}, {'GPU_MODE': 'typo'}, {'GPU_MODE': 'multi_node'},
                    {'ALLOW_BASE_DIT': '1', 'FT_CKPT_PATH': '/checkpoint'},
                    {'ALLOW_BASE_DIT': '1', 'FT_CKPT_IS_FULL': '1'}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_pipeline(**env).returncode, 0)
                self.assertFalse(self.calls.exists())

    def test_explicit_base_renderer_and_custom_checkpoint_preflight_match_selection(self):
        self.assertEqual(self.run_pipeline(ALLOW_BASE_DIT='1').returncode, 0)
        self.assertIn('--allow-base-dit', self.calls.read_text().splitlines()[0])
        self.calls.unlink()
        self.assertEqual(self.run_pipeline(RUN_NAME='custom', FT_CKPT_PATH='/custom/ft.safetensors').returncode, 0)
        self.assertIn('--ft-ckpt /custom/ft.safetensors', self.calls.read_text().splitlines()[0])


if __name__ == '__main__':
    unittest.main()
