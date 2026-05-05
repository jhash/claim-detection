"""Tests for app/cli.py — exercise the argparser without loading a real model."""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest

from app import cli


def test_predict_text_human_output(capsys, fake_predictor):
    with mock.patch("app.cli.Predictor", return_value=fake_predictor):
        rc = cli.main(["predict", "Inflation hit 9.1% last June."])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CLAIM" in out
    assert "confidence" in out


def test_predict_text_json_output(capsys, fake_predictor):
    with mock.patch("app.cli.Predictor", return_value=fake_predictor):
        rc = cli.main(["predict", "--json", "I love this weather"])
    assert rc == 0
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["is_claim"] is False
    assert "confidence" in body


def test_predict_reads_stdin(monkeypatch, capsys, fake_predictor):
    monkeypatch.setattr("sys.stdin", _StringIO("The 2024 budget passed."))
    with mock.patch("app.cli.Predictor", return_value=fake_predictor):
        rc = cli.main(["predict", "-"])
    assert rc == 0
    assert "CLAIM" in capsys.readouterr().out


def test_predict_blank_stdin_returns_2(monkeypatch, capsys, fake_predictor):
    monkeypatch.setattr("sys.stdin", _StringIO("   "))
    with mock.patch("app.cli.Predictor", return_value=fake_predictor):
        rc = cli.main(["predict", "-"])
    assert rc == 2


class _StringIO:
    def __init__(self, s):
        self.s = s

    def read(self) -> str:
        return self.s
