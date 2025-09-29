"""Tests for configuration loading and command‑line interface."""
import tempfile
import os

import numpy as np

from flexipde.io import build_simulation
from flexipde.run import main as run_main


def test_io_appends_suffix(tmp_path):
    # create a temporary TOML configuration
    cfg_text = (
        "[grid]\n"
        "domain = [[0.0, 2.0]]\n"
        "shape = [8]\n"
        "periodic = [true]\n\n"
        "[discretisation]\n"
        "type = \"spectral\"\n"
        "backend = \"numpy\"\n\n"
        "[model]\n"
        "type = \"advection\"\n"
        "[model.parameters]\n"
        "velocity = [1.0]\n\n"
        "[simulation]\n"
        "t0 = 0.0\n"
        "t1 = 0.1\n"
        "dt0 = 0.01\n"
    )
    cfg_path = tmp_path / "testcfg.toml"
    cfg_path.write_text(cfg_text)
    # load with missing suffix
    sim = build_simulation(str(cfg_path.with_suffix("")))
    # check model class
    assert sim.model.__class__.__name__.lower() == "linearadvection"


def test_cli_runs_without_suffix(tmp_path, capsys):
    # create a simple config for diffusion with zero diffusivity
    cfg_text = (
        "[grid]\n"
        "domain = [[0.0, 2.0]]\n"
        "shape = [8]\n"
        "periodic = [true]\n\n"
        "[discretisation]\n"
        "type = \"finite_difference\"\n"
        "backend = \"numpy\"\n\n"
        "[model]\n"
        "type = \"diffusion\"\n"
        "[model.parameters]\n"
        "diffusivity = 0.0\n\n"
        "[simulation]\n"
        "t0 = 0.0\n"
        "t1 = 0.1\n"
        "dt0 = 0.01\n"
        "save_every = 10\n"
    )
    cfg_path = tmp_path / "simple.toml"
    cfg_path.write_text(cfg_text)
    # run CLI
    run_main([str(cfg_path.with_suffix(""))])
    out = capsys.readouterr().out
    assert "Simulation" in out