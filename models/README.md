# Trained models

`surrogate_v1.pt` (1 MB): CNN surrogate from M5, training seed 0, best epoch 109, trained on
Kaggle (Tesla T4) from commit 38ce07f on dataset v1 (5000 cases, seed 0). Test set: Cd MAE 0.049
(2.2 % mean relative error), Cl MAE 0.064. Load with `cfd_drag.predict.Surrogate`
or `cfd_drag.surrogate.train.load_model`. Retrain with `scripts/train.py`.
