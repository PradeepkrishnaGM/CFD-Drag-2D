#!/usr/bin/env bash
# Make nvcc available on a GPU notebook (Kaggle/Colab). Uses the system toolkit if present,
# otherwise downloads NVIDIA's CUDA 12.4 compiler redistributable (driver >= 550 required).
# Prints the directory to add to PATH.
set -euo pipefail
for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
  if [ -x "$d/nvcc" ]; then echo "$d"; exit 0; fi
done
if command -v nvcc >/dev/null; then dirname "$(command -v nvcc)"; exit 0; fi
root="${CUDA_REDIST_DIR:-/tmp/cuda-redist}"
base=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p "$root" && cd "$root"
curl -fsSL "$base/redistrib_12.4.1.json" -o redistrib.json
python3 - "$base" <<'PY' > files.txt
import json, sys
r = json.load(open("redistrib.json"))
for pkg in ("cuda_nvcc", "cuda_cudart", "cuda_crt", "libnvvm"):
    if pkg in r:
        e = r[pkg]["linux-x86_64"]
        print(e["relative_path"], e["sha256"])
PY
while read -r path sha; do
  f=$(basename "$path")
  curl -fsSL "$base/$path" -o "$f"
  echo "$sha  $f" | sha256sum -c --quiet
  tar -xJf "$f" --strip-components=1
  rm "$f"
done < files.txt
ln -sfn lib "$root/lib64"
echo "$root/bin"
