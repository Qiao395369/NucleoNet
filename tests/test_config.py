"""Config flags, presets and reload precedence with fresh FlagValues per test."""

import sys
from absl import flags
import pytest
from vmcnet.train.default_config import get_default_config, get_default_reload_config
from vmcnet.train.parse_config_flags import parse_flags
from vmcnet.utils import io


def parse(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["vmcnet", *map(str, arguments)])
    return parse_flags(flags.FlagValues())


def test_default_flags_and_independent_configs(monkeypatch):
    reload, config = parse(monkeypatch)
    assert reload.to_dict() == get_default_reload_config().to_dict()
    assert config.to_dict() == get_default_config().to_dict()
    assert config.is_locked
    config.gq.h1 = 12
    assert get_default_config().gq.h1 == 64


def test_valid_current_flags_and_tuples(monkeypatch):
    _, config = parse(monkeypatch, "--config.gq.h1=12", "--config.gq.type=ef_shrd_sym_ablation",
                       "--config.problem.ion_pos=((0.,0.,0.),(0.,0.,1.4))", "--config.vmc.acc_steps=3")
    assert config.gq.h1 == 12
    assert config.gq.type == "ef_shrd_sym_ablation"
    assert config.problem.ion_pos == ((0., 0., 0.), (0., 0., 1.4))
    assert config.vmc.acc_steps == 3


def test_unknown_top_level_flag_is_rejected(monkeypatch):
    with pytest.raises(flags.UnrecognizedFlagError):
        parse(monkeypatch, "--not_a_config.value=True")


def test_invalid_reload_field_is_rejected(monkeypatch):
    with pytest.raises(AttributeError):
        parse(monkeypatch, "--reload.logdir_typo=bad")


def test_reload_and_command_line_overrides(monkeypatch, tmp_path):
    cfg = get_default_config()
    cfg.vmc.nepochs = 20
    cfg.gq.h1 = 12
    cfg.base_logdir = str(tmp_path/"base")
    cfg.logdir = str(tmp_path/"old-run")
    io.save_config_dict_to_json(cfg, str(tmp_path), "config")
    reload, restored = parse(monkeypatch, f"--reload.logdir={tmp_path}",
                              "--config.gq.h1=16", "--config.vmc.nburn=4")
    assert restored.vmc.nepochs == 20
    assert restored.gq.h1 == 16
    assert restored.vmc.nburn == 4
    assert restored.logdir == str(tmp_path/"base")
    assert restored.base_logdir == restored.logdir


def test_reload_use_config_file_false_does_not_read_missing_file(monkeypatch, tmp_path):
    _, cfg = parse(monkeypatch, f"--reload.logdir={tmp_path/'missing'}",
                    "--reload.use_config_file=False", "--config.gq.ndet=2")
    assert cfg.gq.ndet == 2


def test_presets_and_reload_are_mutually_exclusive(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="reloaded config"):
        parse(monkeypatch, f"--reload.logdir={tmp_path}", f"--presets.path={tmp_path/'preset.json'}")


def test_preset_file_then_cli_override(monkeypatch, tmp_path):
    cfg = get_default_config()
    cfg.gq.h1 = 12
    cfg.vmc.acc_steps = 2
    io.save_config_dict_to_json(cfg, str(tmp_path), "preset")
    _, restored = parse(monkeypatch, f"--presets.path={tmp_path/'preset.json'}", "--config.gq.h1=16")
    assert restored.gq.h1 == 16
    assert restored.vmc.acc_steps == 2
