## Install

### Dependencies
python3.13


* **GDRCopy**

  Please refer to the [official reposity](https://github.com/NVIDIA/gdrcopy) of GDRCopy.


* **Install DittoCache kernel to sgl-kernel**

```shell

# From repo root
cd sglang-litecache

# Python 3.13 venv
uv python install 3.13
uv venv .venv-py313 --python 3.13
source .venv-py313/bin/activate

# System build deps
apt-get update
apt-get install -y rustc cargo pkg-config libssl-dev

# Newer Rust toolchain for outlines-core / edition2024 deps
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
source /root/.cargo/env

# Python build deps
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
uv pip install -U pip setuptools wheel scikit-build-core cmake ninja setuptools-rust setuptools-scm

# Build sgl-kernel
cd sglang-litecache/sgl-kernel
export CMAKE_ARGS="
  -DSGL_KERNEL_USE_RAFT=ON
  -DSGL_KERNEL_DISABLE_BLACKWELL=ON
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DGDRAPI_LIBRARY=/usr/local/lib/libgdrapi.so
"
# H100
SKIP_FLASHMLA=1 TORCH_CUDA_ARCH_LIST="9.0"  uv pip install -e . --no-build-isolation -v --reinstall -C cmake.define.SGL_KERNEL_COMMON_ARCH=90  -C cmake.define.ENABLE_BELOW_SM90=OFF
# A40
SKIP_FLASHMLA=1 TORCH_CUDA_ARCH_LIST="8.6" uv pip install -e . --no-build-isolation -v --reinstall -C cmake.define.SGL_KERNEL_COMMON_ARCH=86  -C cmake.define.ENABLE_BELOW_SM90=OFF

# Copy compiled editable-install artifacts back to source tree
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

mkdir -p python/sgl_kernel/sm90 python/sgl_kernel/sm100

cp -a "${SITE_PACKAGES}/python/sgl_kernel/sm90/common_ops.abi3.so" \
  python/sgl_kernel/sm90/

cp -a "${SITE_PACKAGES}/python/sgl_kernel/sm100/common_ops.abi3.so" \
  python/sgl_kernel/sm100/

cp -a "${SITE_PACKAGES}/python/sgl_kernel/kvlib_cpu_gather.so" \
  python/sgl_kernel/

cp -a "${SITE_PACKAGES}/sgl_kernel/flash_ops.abi3.so" \
  python/sgl_kernel/

cp -a "${SITE_PACKAGES}/sgl_kernel/spatial_ops.abi3.so" \
  python/sgl_kernel/

# Install sglang with all deps
cd ../python
uv pip install -e ".[all]" --no-build-isolation -v

# Verify
python - <<'PY'
import torch
import sglang
import outlines_core
import sgl_kernel
from sgl_kernel import kvlib
import sgl_kernel.flash_ops
import sgl_kernel.spatial_ops

print("torch", torch.__version__, torch.version.cuda)
print("sglang", sglang.__version__)
print("outlines_core", outlines_core.__version__)
print("sgl_kernel OK")
PY


```


### Build RULER Dataset

Please refer to the [official reposity](https://github.com/NVIDIA/RULER) of RULER.
