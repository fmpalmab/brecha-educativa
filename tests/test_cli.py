"""Pruebas para la interfaz CLI de brecha_educativa."""

import pytest
from brecha_educativa.cli import main


def test_cli_synth():
    exit_code = main(["--synth", "--schools", "10"])
    assert exit_code == 0


def test_cli_report():
    exit_code = main(["--report"])
    assert exit_code == 0
