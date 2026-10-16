"""CPU regression tests for experiment math, data contracts and shell overrides."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from scripts.data.prepare_standard_cl import convert
from src.eval.eval_scl import continual_acc
from src.motivation.common import parse_ratios, unique_runs
from src.motivation.make_lora_ablation_checkpoints import (
    build_ablation, component_scores, drop_rank_components, truncate_svd,
)
from src.motivation.collect_smoke_results import collect_results

ROOT = Path(__file__).resolve().parents[1]


class MotivationTests(unittest.TestCase):
    def test_energy_is_rank_one_frobenius_norm(self):
        a = torch.tensor([[1., 2., 3.], [4., 1., 2.]])
        b = torch.tensor([[1., 4.], [2., 9.], [3., 2.], [7., 1.]])
        expected = torch.stack([torch.linalg.norm(torch.outer(b[:, i], a[i])) for i in range(2)])
        torch.testing.assert_close(component_scores(a, b), expected)

    def test_svd_matches_dense_truncation_and_preserves_scaling(self):
        torch.manual_seed(12)
        a, b = torch.randn(4, 7), torch.randn(9, 4)
        u, s, vh = torch.linalg.svd(b @ a, full_matrices=False)
        for drop in (0, 1, 3, 4):
            aa, bb = truncate_svd(a, b, drop)
            keep = 4 - drop
            expected = (u[:, :keep] * s[:keep]) @ vh[:keep]
            torch.testing.assert_close(2 * bb @ aa, 2 * expected, atol=1e-5, rtol=1e-5)
            self.assertEqual(aa.shape, a.shape)
            self.assertEqual(bb.shape, b.shape)

    def test_drop_identity_full_and_random_nested(self):
        state = {"layer.lora_A.weight": torch.randn(8, 12),
                 "layer.lora_B.weight": torch.randn(10, 8)}
        for selection in ("energy_low", "svd_low", "random"):
            same, _ = drop_rank_components(state, 0, selection, 42)
            for key in state:
                self.assertTrue(torch.equal(same[key], state[key]))
            zero, _ = drop_rank_components(state, 1, selection, 42)
            self.assertEqual(torch.count_nonzero(zero["layer.lora_B.weight"] @ zero["layer.lora_A.weight"]), 0)
        _, small = drop_rank_components(state, .25, "random", 42)
        _, large = drop_rank_components(state, .5, "random", 42)
        self.assertTrue(set(small["layer.lora_A.weight"]) <= set(large["layer.lora_A.weight"]))
        for invalid in ("nan", "inf", "-0.1", "1.1", ""):
            with self.assertRaises(ValueError):
                parse_ratios(invalid)

    def test_data_contract_reaches_scl_metric(self):
        labels = ["very negative", "negative", "neutral", "positive", "very positive"]
        rows = convert([{"sentence": "Title: sample\nText: good\n", "label": "positive"}], labels, "sentiment")
        self.assertTrue(rows[0]["conversations"][0]["value"].endswith("\nAnswer:"))
        self.assertEqual(continual_acc({"text": "positive", "additional_info": rows[0]["eval"]}), 1)
        self.assertEqual(continual_acc({"text": "negative", "additional_info": rows[0]["eval"]}), 0)
        with self.assertRaises(ValueError):
            convert([{"sentence": "text", "label": "unknown"}], labels, "sentiment")

    def test_grid_checkpoints_and_scl_collection(self):
        modes = ["seq_lora", "drop_first", "drop_second", "drop_all"]
        self.assertEqual(len(list(unique_runs(modes, [0, .25, .5]))), 7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, logs = root / "source", root / "ablated", root / "logs"
            for order in (1, 2):
                directory = source / f"order{order}"
                directory.mkdir(parents=True)
                save_file({"layer.lora_A.weight": torch.randn(4, 5),
                           "layer.lora_B.weight": torch.randn(6, 4)}, str(directory / "adapter_model.safetensors"))
                (directory / "adapter_config.json").write_text('{"r":4,"lora_alpha":8}')
            build_ablation(source, output, [0, .5], modes, "svd_low", 42)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["runs"]), 4)
            for run in manifest["runs"]:
                directory = logs / run["mode"] / f'r{run["ratio"]:g}' / "amazon"
                directory.mkdir(parents=True)
                (directory / "eval.log").write_text("Acc in amazon: 0.75\n")
            rows = collect_results(logs, "amazon", modes, [0, .5])
            self.assertTrue(all(row["accuracy"] == .75 for row in rows))
            self.assertTrue((output / "drop_second/r0.5/order1").is_symlink())
            self.assertEqual((output / "drop_all/r0.5/order1").resolve(),
                             (output / "drop_first/r0.5/order1").resolve())
            original = load_file(str(source / "order2/adapter_model.safetensors"))
            changed = load_file(str(output / "drop_second/r0.5/order2/adapter_model.safetensors"))
            self.assertFalse(torch.equal(original["layer.lora_A.weight"], changed["layer.lora_A.weight"]))
            env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": sys.executable,
                   "ABLATION_ROOT": str(output), "LOG_ROOT": str(logs)}
            evaluation = subprocess.run(["bash", "scripts/motivation/run.sh", "eval"],
                                        cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(evaluation.stdout.count("-m src.eval.model_diverse_gen_batch"), 4)
            self.assertEqual(evaluation.stdout.count("-m src.eval.eval_scl"), 4)
            summary = root / "summary"
            subprocess.run([sys.executable, "-m", "src.motivation.collect_smoke_results",
                            "--log-root", str(logs), "--manifest", str(output / "manifest.json"),
                            "--output-dir", str(summary)], cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(len(json.loads((summary / "results.json").read_text())), 8)
            self.assertTrue((summary / "figure_task1_accuracy.png").is_file())
            with self.assertRaises(FileExistsError):
                build_ablation(source, output, [.5], modes, "svd_low", 42)

    def test_environment_overrides_reach_training_command(self):
        env = {**os.environ, "DRY_RUN": "1", "MAX_STEPS": "20", "LORA_R": "16",
               "LORA_ALPHA": "32", "TRAIN_EPOCHS": "2", "PRECISION": "fp16"}
        result = subprocess.run(["bash", "scripts/motivation/run_two_task_seq_lora_smoke.sh"],
                                cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        for flag in ("--max_steps 20", "--lora_r 16", "--lora_alpha 32", "--num_train_epochs 2", "--fp16 True"):
            self.assertIn(flag, result.stdout)
        self.assertEqual(result.stdout.count("src/train/cl_train.py"), 2)


if __name__ == "__main__":
    unittest.main()
