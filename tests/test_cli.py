"""CLI arguments and binary allocation checks."""

import argparse
import importlib

import pytest

from fastmri_preprocess.paths import add_output_argument
from fastmri_preprocess.run import preallocate


@pytest.mark.parametrize(
    "module",
    ["survey", "run", "finalize", "repair_holes", "sweep", "merge_sweep", "verify"],
)
def test_cli_help_without_dataset_or_gpu(module, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [module, "--help"])
    with pytest.raises(SystemExit) as result:
        importlib.import_module(f"fastmri_preprocess.{module}").main()
    assert result.value.code == 0
    assert "--out" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_output_path_environment_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTMRI_OUT_ROOT", str(tmp_path))
    parser = argparse.ArgumentParser()
    add_output_argument(parser)
    assert parser.parse_args([]).out == tmp_path
    assert parser.parse_args(["--out", str(tmp_path / "other")]).out == (
        tmp_path / "other"
    )


def test_preallocation_preserves_existing_files(tmp_path):
    preallocate(tmp_path, n_slices=1, n_scans=1)
    path = tmp_path / "gt.bin"
    with path.open("r+b") as stream:
        stream.write(b"existing data")
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }
    preallocate(tmp_path, n_slices=1, n_scans=1)
    assert before == {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }
    assert path.read_bytes().startswith(b"existing data")
    with pytest.raises(ValueError, match="Refusing to resize"):
        preallocate(tmp_path, n_slices=2, n_scans=1)
    assert path.read_bytes().startswith(b"existing data")


def test_preallocation_checks_all_files_before_creating_any(tmp_path):
    path = tmp_path / "pe_mask.bin"
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="Refusing to resize"):
        preallocate(tmp_path, n_slices=1, n_scans=1)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == b"keep"
