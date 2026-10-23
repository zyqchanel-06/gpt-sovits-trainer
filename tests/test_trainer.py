# -*- coding: utf-8 -*-
"""训练器单测：全部 dry-run，不真的跑训练。

覆盖：配置探测/校验、状态机断点续跑、六步命令拼装、产物扫描、网关写入。
跑法：python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sovits_trainer import ConfigError, STAGES, Task, TrainerConfig  # noqa: E402
from sovits_trainer import gateway as gateway_mod  # noqa: E402
from sovits_trainer.stages import asr, prep, register, s1, s2, slice as slice_stage  # noqa: E402


class FakeGptSovits:
    """造一个最小的 GPT-SoVITS 目录结构，只用于拼命令、不执行。"""

    def __init__(self, base: Path):
        self.base = base
        base.mkdir(parents=True, exist_ok=True)
        (base / "webui.py").write_text("# fake", encoding="utf-8")
        gsv = base / "GPT_SoVITS"
        (gsv / "configs").mkdir(parents=True, exist_ok=True)
        (gsv / "pretrained_models").mkdir(parents=True, exist_ok=True)
        (gsv / "s2_train.py").write_text("# fake", encoding="utf-8")
        (gsv / "s1_train.py").write_text("# fake", encoding="utf-8")
        (gsv / "configs" / "s2.json").write_text(json.dumps({
            "train": {"epochs": 100, "batch_size": 32, "fp16_run": True},
            "data": {"exp_dir": "", "max_wav_value": 32768.0},
            "model": {"version": "v2"},
            "s2_ckpt_dir": "",
        }), encoding="utf-8")
        (gsv / "configs" / "s1longer-v2.yaml").write_text(
            "train:\n  epochs: 20\n  batch_size: 8\n  precision: 16-mixed\n"
            "data:\n  num_workers: 4\nmodel:\n  vocab_size: 1025\n", encoding="utf-8")
        tools = base / "tools"
        (tools / "asr").mkdir(parents=True, exist_ok=True)
        (tools / "slice_audio.py").write_text("# fake", encoding="utf-8")
        (tools / "asr" / "fasterwhisper_asr.py").write_text("# fake", encoding="utf-8")
        (tools / "asr" / "funasr_asr.py").write_text("# fake", encoding="utf-8")


class BaseCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.fake = FakeGptSovits(self.tmp / "gpt-sovits")
        self.config = TrainerConfig(
            gpt_sovits_root=self.fake.base,
            work_root=self.tmp / "tasks",
            python=sys.executable,
        )

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------- 配置
class TestConfig(BaseCase):
    def test_validate_accepts_fake_tree(self):
        self.config.validate()  # 不抛就算过

    def test_validate_rejects_non_gpt_sovits(self):
        bad = TrainerConfig(gpt_sovits_root=self.tmp / "not-gsv", work_root=self.tmp / "t")
        with self.assertRaises(ConfigError):
            bad.validate()

    def test_default_paths(self):
        self.assertEqual(
            self.config.weights_dir,
            (self.fake.base / "GPT_SoVITS" / "pretrained_models").resolve(),
        )
        self.assertEqual(self.config.stage_dir("v1", "slice"), (self.tmp / "tasks" / "v1" / "slice"))

    def test_roundtrip(self):
        data = self.config.as_dict()
        restored = TrainerConfig.from_dict(data)
        self.assertEqual(restored.gpt_sovits_root, self.config.gpt_sovits_root)
        self.assertEqual(restored.version, "v2")

    def test_load_from_env(self):
        old = os.environ.get("GPT_SOVITS_ROOT")
        os.environ["GPT_SOVITS_ROOT"] = str(self.fake.base)
        try:
            loaded = TrainerConfig.load()
            self.assertEqual(loaded.gpt_sovits_root, self.fake.base.resolve())
        finally:
            if old is None:
                os.environ.pop("GPT_SOVITS_ROOT", None)
            else:
                os.environ["GPT_SOVITS_ROOT"] = old


# ---------------------------------------------------------------- 状态机
class TestState(BaseCase):
    def test_create_and_persist(self):
        task = Task(voice="alice", root=self.config.work_root)
        task.save()
        loaded = Task.load(self.config.work_root, "alice")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.voice, "alice")
        self.assertEqual(set(loaded.stages), set(STAGES))

    def test_resume_skips_done(self):
        task = Task(voice="alice", root=self.config.work_root)
        task.start("slice")
        task.finish("slice", "/tmp/wav")
        task.save()
        loaded = Task.load(self.config.work_root, "alice")
        self.assertTrue(loaded.is_done("slice"))
        self.assertEqual(loaded.next_pending(), "asr")

    def test_fail_records_error(self):
        task = Task(voice="alice", root=self.config.work_root)
        task.start("asr")
        task.fail("asr", "模型没下载")
        self.assertEqual(task.stages["asr"].status, "failed")
        self.assertIn("模型没下载", task.stages["asr"].error)
        # 失败的步骤不算完成，下次还会重试
        self.assertFalse(task.is_done("asr"))

    def test_skip_counts_as_done(self):
        task = Task(voice="alice", root=self.config.work_root)
        task.skip("asr", "用现成标注")
        self.assertTrue(task.is_done("asr"))

    def test_reset_from(self):
        task = Task(voice="alice", root=self.config.work_root)
        for stage in STAGES[:4]:
            task.start(stage)
            task.finish(stage)
        reset = task.reset_from("s2")
        self.assertEqual(reset, ["s2", "s1", "register"])
        self.assertTrue(task.is_done("prep"))
        self.assertEqual(task.stages["s2"].status, "pending")


# ---------------------------------------------------------------- 命令拼装
class TestCommands(BaseCase):
    def test_slice_command_order(self):
        cmd = slice_stage.build_command(
            self.config, Path("in.wav"), Path("out"),
            threshold=-34, min_length=4000, min_interval=300,
            hop_size=10, max_sil_kept=500, max_length=10000, alpha=0.25,
        )
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[2], "tools/slice_audio.py")
        self.assertEqual(cmd[3], "in.wav")
        self.assertEqual(cmd[4], "out")
        # 9 个位置参数：threshold min_length min_interval hop_size max_sil_kept max _max alpha i_part all_part
        self.assertEqual(cmd[5:], ["-34", "4000", "300", "10", "500", "10000", "0.25", "0", "1"])

    def test_asr_command(self):
        cmd = asr.build_command(
            self.config, Path("in"), Path("out"),
            backend="faster-whisper", model_size="large-v3", language="zh", precision="float16",
        )
        self.assertIn("tools/asr/fasterwhisper_asr.py", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], "in")
        self.assertEqual(cmd[cmd.index("-o") + 1], "out")
        self.assertEqual(cmd[cmd.index("-l") + 1], "zh")

    def test_asr_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            asr.build_command(self.config, Path("i"), Path("o"), backend="whisperx")

    def test_prep_env(self):
        task = Task(voice="alice", root=self.config.work_root)
        env = prep.build_env(
            self.config, task, Path("a.list"), Path("wav"), Path("exp"),
        )
        self.assertEqual(env["inp_text"], "a.list")
        self.assertEqual(env["inp_wav_dir"], "wav")
        self.assertEqual(env["opt_dir"], "exp")
        self.assertEqual(env["exp_name"], "alice")
        self.assertEqual(env["i_part"], "0")
        self.assertEqual(env["all_parts"], "1")
        self.assertEqual(env["version"], "v2")
        self.assertTrue(env["bert_pretrained_dir"].endswith("chinese-roberta-wwm-ext-large"))

    def test_s2_config_overrides(self):
        task = Task(voice="alice", root=self.config.work_root)
        data = s2.build_config(self.config, task, epochs=4, batch_size=2)
        self.assertEqual(data["train"]["epochs"], 4)
        self.assertEqual(data["train"]["batch_size"], 2)
        self.assertEqual(data["name"], "alice")
        self.assertEqual(data["save_weight_dir"], str(self.config.weights_dir))
        self.assertIn("gsv-v2final-pretrained", data["train"]["pretrained_s2G"])

    def test_s2_config_fp32_halves_batch(self):
        task = Task(voice="alice", root=self.config.work_root)
        cfg = TrainerConfig(
            gpt_sovits_root=self.fake.base, work_root=self.tmp / "tasks",
            python=sys.executable, is_half=False,
        )
        data = s2.build_config(cfg, task, batch_size=4)
        self.assertFalse(data["train"]["fp16_run"])
        self.assertEqual(data["train"]["batch_size"], 2)

    def test_s1_config_requires_yaml(self):
        task = Task(voice="alice", root=self.config.work_root)
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("没装 PyYAML")
        data = s1.build_config(self.config, task, epochs=3, batch_size=1)
        self.assertEqual(data["train"]["epochs"], 3)
        self.assertEqual(data["train"]["exp_name"], "alice")
        self.assertTrue(data["train_semantic_path"].endswith("6-name2semantic.tsv"))
        self.assertTrue(data["train_phoneme_path"].endswith("2-name2text.txt"))


# ---------------------------------------------------------------- 产物登记
class TestRegister(BaseCase):
    def _touch(self, name: str, size: int = 1024) -> Path:
        path = self.config.weights_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def test_scan_finds_both_kinds(self):
        self._touch("alice_e4_s124.pth")
        self._touch("alice_e8_s248.pth")
        self._touch("alice-e4.ckpt")
        self._touch("alice-e8.ckpt")
        self._touch("bob_e8_s100.pth")  # 别的声线，不该算进来
        found = register.scan(self.config.weights_dir, "alice")
        self.assertEqual([item["epoch"] for item in found["sovits"]], [4, 8])
        self.assertEqual([item["epoch"] for item in found["gpt"]], [4, 8])

    def test_pick_best_prefers_latest_epoch(self):
        self._touch("alice_e4_s124.pth")
        self._touch("alice_e8_s248.pth")
        self._touch("alice-e4.ckpt")
        self._touch("alice-e8.ckpt")
        chosen = register.pick_best(register.scan(self.config.weights_dir, "alice"))
        self.assertEqual(chosen["epoch"], 8)
        self.assertTrue(chosen["sovits"].endswith("alice_e8_s248.pth"))

    def test_register_writes_voices_file(self):
        self._touch("alice_e8_s248.pth")
        self._touch("alice-e8.ckpt")
        task = Task(voice="alice", root=self.config.work_root)
        outcome = register.run(self.config, task)
        payload = json.loads(Path(outcome["voices_file"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["voice"], "alice")
        self.assertEqual(payload["epoch"], 8)

    def test_register_fails_when_missing(self):
        task = Task(voice="ghost", root=self.config.work_root)
        with self.assertRaises(RuntimeError):
            register.run(self.config, task)


# ---------------------------------------------------------------- 网关对接
class TestGateway(BaseCase):
    def _gateway_config(self) -> Path:
        path = self.tmp / "gateway.json"
        path.write_text(json.dumps({
            "port": 9881,
            "voices": [{"id": "existing", "name": "Existing", "refAudio": "a.wav",
                        "promptText": "t", "promptLang": "zh", "gpt": "", "sovits": ""}],
        }), encoding="utf-8")
        return path

    def test_add_new_voice(self):
        path = self._gateway_config()
        result = gateway_mod.upsert_voice(
            path, voice_id="alice", sovits_path="a.pth", gpt_path="a.ckpt",
            ref_audio="ref.wav", prompt_text="参考文本",
        )
        self.assertEqual(result["action"], "added")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["voices"]), 2)
        self.assertEqual(data["voices"][1]["sovits"], "a.pth")

    def test_update_existing_voice(self):
        path = self._gateway_config()
        gateway_mod.upsert_voice(
            path, voice_id="existing", sovits_path="new.pth", gpt_path="new.ckpt",
            ref_audio="ref.wav", prompt_text="新文本",
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["voices"]), 1)
        self.assertEqual(data["voices"][0]["sovits"], "new.pth")
        self.assertEqual(data["voices"][0]["promptText"], "新文本")

    def test_backup_created(self):
        path = self._gateway_config()
        result = gateway_mod.upsert_voice(
            path, voice_id="alice", sovits_path="a.pth", gpt_path="a.ckpt",
            ref_audio="r", prompt_text="t",
        )
        self.assertTrue(Path(result["backup"]).exists())

    def test_read_first_sample(self):
        list_file = self.tmp / "a.list"
        list_file.write_text("D:/w/1.wav|alice|zh|你好世界\nD:/w/2.wav|alice|zh|第二句\n",
                             encoding="utf-8")
        sample = gateway_mod.read_first_sample(list_file)
        self.assertEqual(sample["wav"], "D:/w/1.wav")
        self.assertEqual(sample["text"], "你好世界")
        self.assertEqual(sample["lang"], "zh")

    def test_rejects_bad_json(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(gateway_mod.GatewayError):
            gateway_mod.upsert_voice(path, voice_id="x", sovits_path="a", gpt_path="b",
                                     ref_audio="r", prompt_text="t")


if __name__ == "__main__":
    unittest.main()
