"""The command-line help must not start a calculation."""

import sys

from vmcnet.train import runners


def test_help_exits_before_config_parsing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["vmc-molecule", "--help"])

    def must_not_parse(_):
        raise AssertionError("help must not launch a calculation")

    monkeypatch.setattr(runners.train.parse_config_flags, "parse_flags", must_not_parse)
    runners.run_molecule()
    assert "Usage: vmc-molecule" in capsys.readouterr().out
