# Trained models

`surrogate_v1.pt` (1 MB): CNN surrogate from M5, training seed 0, best epoch 109, trained on
Kaggle (Tesla T4) from commit 38ce07f on dataset v1 (5000 cases, seed 0). Test set: Cd MAE 0.049
(2.2 % mean relative error), Cl MAE 0.064. Load with `cfd_drag.predict.Surrogate`
or `cfd_drag.surrogate.train.load_model`. Retrain with `scripts/train.py`.

`field_unet_v1.pt` (31 MB, not in git): U-Net flow-field surrogate from M8, training seed 0, best
epoch 92, trained on Kaggle (Tesla T4) from commit 59eb5e1 on the v2 field dataset. Test set:
velocity relative L2 error 1.93 %, Cp 5.08 %. Download it with `scripts/fetch_field_model.py`
(GitHub release v2.0); load with `cfd_drag.predict.FieldSurrogate` or
`cfd_drag.surrogate.field_train.load_field_model`. Retrain with `scripts/train_fields.py`.
