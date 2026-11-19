# Trained models

`surrogate_v2.pt` (1 MB): CNN surrogate from M10, same architecture as v1, trained with mirror
(flip) augmentation, training seed 0, best epoch 144, on Kaggle (Tesla T4) from commit a8904c1 on
dataset v1 (5000 cases, seed 0). Test set: Cd MAE 0.044 (2.0 % mean relative error), Cl MAE 0.044;
mean |Cd - Cd(mirrored mask)| 0.024. Load with `cfd_drag.predict.Surrogate` or
`cfd_drag.surrogate.train.load_model`. Retrain with `scripts/train.py --flip`.
It replaces `surrogate_v1.pt` (M5, no augmentation: Cd MAE 0.049, Cl MAE 0.064), which is in the
git history and the v1.0 / v2.0 releases.

`field_unet_v1.pt` (31 MB, not in git): U-Net flow-field surrogate from M8, training seed 0, best
epoch 92, trained on Kaggle (Tesla T4) from commit 59eb5e1 on the v2 field dataset. Test set:
velocity relative L2 error 1.93 %, Cp 5.08 %. Download it with `scripts/fetch_field_model.py`
(GitHub release v2.0); load with `cfd_drag.predict.FieldSurrogate` or
`cfd_drag.surrogate.field_train.load_field_model`. Retrain with `scripts/train_fields.py`.
