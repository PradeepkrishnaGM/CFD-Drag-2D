"""Download the trained U-Net field model (31 MB) from the GitHub release into models/.

Example:
    python scripts/fetch_field_model.py

The CNN for Cd/Cl (1 MB) is tracked in git; the U-Net weights are a release asset to keep the
repository small. Flow Studio works without them (no field prediction).
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.predict import FIELD_MODEL_PATH  # noqa: E402

URL = "https://github.com/PradeepkrishnaGM/CFD-Drag-2D/releases/download/v2.0/field_unet_v1.pt"


def main():
    if FIELD_MODEL_PATH.exists():
        print(f"{FIELD_MODEL_PATH} already exists")
        return
    FIELD_MODEL_PATH.parent.mkdir(exist_ok=True)
    tmp = FIELD_MODEL_PATH.with_suffix(".part")
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(FIELD_MODEL_PATH)
    print(f"saved {FIELD_MODEL_PATH} ({FIELD_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
