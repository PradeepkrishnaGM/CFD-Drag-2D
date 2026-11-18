"""CNN surrogate: split invariants, model wiring, determinism, train/evaluate smoke run."""
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from cfd_drag.dataset import load  # noqa: E402
from cfd_drag.params import TRAIN  # noqa: E402
from cfd_drag.paths import REPO_ROOT  # noqa: E402
from cfd_drag.surrogate.data import Normaliser, split_indices, tabular_features  # noqa: E402
from cfd_drag.surrogate.model import SurrogateCNN  # noqa: E402
from cfd_drag.surrogate.train import train  # noqa: E402

EXAMPLE = REPO_ROOT / "example_data"
TINY = replace(TRAIN, channels=(8, 16), hidden=16, epochs=3, batch_size=8)


def test_split_is_disjoint_complete_and_seeded():
    s = split_indices(5000)
    assert [len(s[k]) for k in ["train", "val", "test"]] == [3500, 750, 750]
    allidx = np.concatenate(list(s.values()))
    assert len(np.unique(allidx)) == 5000 == len(allidx)
    s2 = split_indices(5000)
    assert all(np.array_equal(s[k], s2[k]) for k in s)
    s3 = split_indices(5000, replace(TRAIN, split_seed=1))
    assert not np.array_equal(s["test"], s3["test"])


def test_normaliser_round_trip():
    df, _ = load(EXAMPLE)
    norm = Normaliser.fit(df)
    assert np.allclose(norm.y(df).mean(axis=0), 0, atol=1e-12)
    assert np.allclose(norm.y_inverse(norm.y(df)), df[["Cd", "Cl"]].to_numpy())
    assert tabular_features(df).shape == (len(df), 12)


def test_model_shapes_and_re_is_used():
    torch.manual_seed(0)
    model = SurrogateCNN(TINY).eval()
    mask = (torch.rand(4, 1, 64, 64) > 0.8).float()
    out_a = model(mask, torch.zeros(4, 1))
    out_b = model(mask, torch.ones(4, 1))
    assert out_a.shape == (4, 2)
    assert not torch.allclose(out_a, out_b)
    assert sum(p.numel() for p in SurrogateCNN().parameters()) < 400_000


def test_training_is_deterministic_on_cpu(tmp_path):
    df, masks = load(EXAMPLE)
    a = train(df, masks, tmp_path / "a", TINY, "cpu", log=lambda *_: None)
    b = train(df, masks, tmp_path / "b", TINY, "cpu", log=lambda *_: None)
    assert a["best_val_cd_mae"] == b["best_val_cd_mae"]
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])


def test_train_and_evaluate_scripts(tmp_path):
    run = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/train.py", "--data", str(EXAMPLE), "--out", str(run),
                    "--epochs", "1", "--device", "cpu"], cwd=REPO_ROOT, check=True, capture_output=True)
    assert (run / "model.pt").exists() and (run / "split.json").exists()
    # one epoch will not beat the baselines: exit code 1 is fine, outputs must exist
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--data", str(EXAMPLE), "--run", str(run)],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    metrics = json.loads((run / "metrics.json").read_text())
    assert {"cnn", "gbm", "mean"} <= metrics.keys()
    assert (run / "predictions.csv").exists() and (run / "parity.png").exists()
