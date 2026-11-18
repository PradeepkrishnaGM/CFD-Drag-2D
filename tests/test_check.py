"""Dataset checker: passes on the example set, catches each kind of broken dataset."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from cfd_drag.check import check, check_dir
from cfd_drag.paths import REPO_ROOT

EXAMPLE = REPO_ROOT / "example_data"


@pytest.fixture
def example():
    df = pd.read_csv(EXAMPLE / "data.csv", keep_default_na=False, na_values=["nan", "NaN"])
    masks = np.load(EXAMPLE / "masks.npy")
    config = json.loads((EXAMPLE / "config.json").read_text())
    return df, masks, config


def test_example_set_passes(example, tmp_path):
    summary, problems = check(*example)
    assert problems == []
    assert summary["passed"] and summary["n"] == summary["n_ok"] == 24
    # check_dir writes summary.json without touching the source files
    for name in ["data.csv", "masks.npy", "config.json"]:
        shutil.copy(EXAMPLE / name, tmp_path / name)
    summary2, _ = check_dir(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary2


def _fails(df, masks, config, text):
    _, problems = check(df, masks, config)
    assert any(text in p for p in problems), problems


def test_low_ok_fraction(example):
    df, masks, config = example
    df = df.copy()
    df.loc[:1, "ok"] = False  # 22/24 = 0.917 < 0.95
    _fails(df, masks, config, "ok fraction")


def test_non_finite_target(example):
    df, masks, config = example
    df = df.copy()
    df.loc[3, "Cd"] = np.nan
    _fails(df, masks, config, "non-finite")


def test_duplicate_inputs(example):
    df, masks, config = example
    df = df.copy()
    df.loc[5, ["shape", "size1", "size2", "angle", "re"]] = df.loc[4, ["shape", "size1", "size2", "angle", "re"]]
    _fails(df, masks, config, "duplicate")


def test_mask_problems(example):
    df, masks, config = example
    _fails(df, masks[:-1], config, "masks for")
    empty = masks.copy()
    empty[0] = 0
    _fails(df, empty, config, "empty mask")


def test_range_problems(example):
    df, masks, config = example
    df = df.copy()
    df.loc[0, "re"] = config["sampling"]["re_max"] + 1
    df.loc[1, "blockage"] = config["physics"]["max_blockage"] + 0.01
    _, problems = check(df, masks, config)
    assert any("Re outside" in p for p in problems)
    assert any("blockage" in p for p in problems)


@pytest.mark.slow
def test_spot_check_reproduces_example_set(solver_exe, tmp_path):
    from cfd_drag.check import spot_check
    for name in ["data.csv", "masks.npy", "config.json"]:
        shutil.copy(EXAMPLE / name, tmp_path / name)
    res = spot_check(tmp_path, k=2, workers=2)
    assert res["passed"], res
    # a tampered target is caught
    df = pd.read_csv(tmp_path / "data.csv", keep_default_na=False)
    df.loc[res["indices"], "Cd"] *= 1.01
    df.to_csv(tmp_path / "data.csv", index=False)
    assert not spot_check(tmp_path, k=2, workers=2)["passed"]
