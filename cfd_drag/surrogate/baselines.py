"""Baselines from SPEC.md: mean predictor and gradient boosting on shape parameters."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..params import TRAIN, TrainParams
from .data import TARGETS, tabular_features
from .metrics import report


def run_baselines(df: pd.DataFrame, split: dict, params: TrainParams = TRAIN) -> dict:
    """Fit on the train rows, return test metrics for each baseline."""
    train, test = df.iloc[split["train"]], df.iloc[split["test"]]
    y_train = train[TARGETS].to_numpy(float)
    results = {"mean": report(test, np.tile(y_train.mean(axis=0), (len(test), 1)))}

    x_train, x_test = tabular_features(train), tabular_features(test)
    pred = np.zeros((len(test), len(TARGETS)))
    for j in range(len(TARGETS)):
        gbm = HistGradientBoostingRegressor(max_iter=params.gbm_max_iter,
                                            learning_rate=params.gbm_learning_rate,
                                            random_state=params.seed)
        gbm.fit(x_train, y_train[:, j])
        pred[:, j] = gbm.predict(x_test)
    results["gbm"] = report(test, pred)
    return results
