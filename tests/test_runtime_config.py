"""Weekly sector job writes to config/runtime/, readers fall back to the
tracked seed (2026-11-02).  Before, the job rewrote tracked files and every
VPS ``git pull`` stopped on a dirty checkout."""
from __future__ import annotations

import json

import storage.database as dbmod
import scripts.sector_correlation as sc


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _point_db_at(tmp_path, monkeypatch):
    seed = tmp_path / "config" / "sector_map.json"
    runtime = tmp_path / "config" / "runtime" / "sector_map.json"
    monkeypatch.setattr(dbmod, "_SECTOR_MAP_PATH", str(seed))
    monkeypatch.setattr(dbmod, "_SECTOR_MAP_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(dbmod, "_SECTOR_MAP_CACHE", None)
    monkeypatch.setattr(dbmod, "_SECTOR_MAP_MTIME", None)
    monkeypatch.setattr(dbmod, "_SECTOR_MAP_SOURCE", None)
    return seed, runtime


def test_reader_uses_seed_until_runtime_exists(tmp_path, monkeypatch):
    seed, runtime = _point_db_at(tmp_path, monkeypatch)
    _write(seed, {"CVX": {"sector": "energy_seed", "peers": ["XOM"]}})
    assert dbmod.get_sector("CVX") == "energy_seed"
    _write(runtime, {"CVX": {"sector": "energy", "peers": ["XOM", "TRGP"]}})
    assert dbmod.get_sector("CVX") == "energy"
    assert dbmod.get_peers("cvx") == ["XOM", "TRGP"]


def test_reader_without_any_file_is_empty(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    assert dbmod.get_sector("CVX") is None
    assert dbmod.get_peers("CVX") == []


def test_job_writes_runtime_not_seed(tmp_path, monkeypatch):
    seed = tmp_path / "config" / "sector_map.json"
    runtime_dir = tmp_path / "config" / "runtime"
    monkeypatch.setattr(sc, "SECTOR_MAP_SEED_PATH", seed)
    monkeypatch.setattr(sc, "SECTOR_MAP_PATH", runtime_dir / "sector_map.json")
    monkeypatch.setattr(sc, "CORRELATION_MATRIX_PATH", runtime_dir / "correlation_matrix.json")
    _write(seed, {"CVX": {"sector": "energy"}})

    loaded = sc.load_sector_map()          # first run: from the seed
    assert loaded == {"CVX": {"sector": "energy"}}
    loaded["CVX"]["peers"] = ["XOM"]
    sc.save_sector_map(loaded)
    sc.save_correlation_matrix({"CVX": {"XOM": 0.8}})

    assert json.loads(seed.read_text()) == {"CVX": {"sector": "energy"}}  # seed untouched
    assert json.loads((runtime_dir / "sector_map.json").read_text())["CVX"]["peers"] == ["XOM"]
    assert (runtime_dir / "correlation_matrix.json").exists()
    assert sc.load_sector_map()["CVX"]["peers"] == ["XOM"]  # next run reads runtime


def test_runtime_dir_is_gitignored():
    from pathlib import Path
    gi = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert "config/runtime/" in gi
