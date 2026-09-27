# DittoCache

## Install

### Dependencies
python3.13


* **GDRCopy**

  GDRCopy is required for DittoCache. Install it on the host machine before building `sgl-kernel`; the B1-B6 speedup ablation profiles use the `gdrcopy` transfer backend, and the kernel build below links against `libgdrapi.so`.

  GDRCopy installation needs root/sudo privileges and a non-container host environment that can build and load the `gdrdrv` kernel module. Installing only inside an unprivileged Docker/container runtime is not enough unless the host already provides the GDRCopy driver and runtime library.

  Please refer to the [official repository](https://github.com/NVIDIA/gdrcopy) for installation. After installation, verify the host has the driver and library visible:

  ```shell
  lsmod | grep gdrdrv
  test -e /dev/gdrdrv && echo "gdrdrv device OK"
  ldconfig -p | grep libgdrapi
  ```

  The build command below assumes `libgdrapi.so` is available at `/usr/local/lib/libgdrapi.so`; adjust `-DGDRAPI_LIBRARY` if your installation places it elsewhere.


* **Install DittoCache kernel to sgl-kernel**

```shell

# Enter the repository
cd DittoCache

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
cd sgl-kernel
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

cp -a "${SITE_PACKAGES}/sgl_kernel/sm90/common_ops.abi3.so" \
  python/sgl_kernel/sm90/

cp -a "${SITE_PACKAGES}/sgl_kernel/sm100/common_ops.abi3.so" \
  python/sgl_kernel/sm100/

cp -a "${SITE_PACKAGES}/sgl_kernel/kvlib_cpu_gather.so" \
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

## Data and Models

The scripts use `/models` and `/datasets` by default. You can keep these paths or override them with environment variables such as `MODEL_PATH`, `DATASET_PATH`, `LONGBENCH_PATH`, and `LONGBENCH_V2_PATH`. Download sources: [Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M), [Llama-3-8B-Instruct-Gradient-1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k), [LongBench](https://huggingface.co/datasets/THUDM/LongBench), [LongBench-v2](https://huggingface.co/datasets/THUDM/LongBench-v2), and [RULER](https://github.com/NVIDIA/RULER).

### Model Downloads

```shell
# Hugging Face access. Login is needed for gated or private repos.
uv pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login

mkdir -p /models

# Qwen long-context model used by the Qwen configs/scripts.
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-1M --local-dir /models/Qwen2.5-14B-Instruct-1M

# Llama long-context model used by the Llama configs/scripts.
huggingface-cli download gradientai/Llama-3-8B-Instruct-Gradient-1048k --local-dir /models/Llama-3-8B-Instruct-Gradient-1048k

# Some Llama scripts default to /models/Llama-3-8B-Instruct.
ln -sfn /models/Llama-3-8B-Instruct-Gradient-1048k /models/Llama-3-8B-Instruct
```

When preparing hash weights, match the model shape parameters to the selected model:

| Model | `MODEL_PATH` | `NUM_LAYERS` | `NUM_HEADS` | `NUM_KV_HEADS` | `HEAD_DIM` |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct-1M | `/models/Qwen2.5-14B-Instruct-1M` | 48 | 40 | 8 | 128 |
| Llama-3-8B-Instruct-Gradient-1048k | `/models/Llama-3-8B-Instruct-Gradient-1048k` | 32 | 32 | 8 | 128 |

### Dataset Downloads

```shell
mkdir -p /datasets

# LongBench. Ditto scripts read the task files from /datasets/LongBench/data.
huggingface-cli download THUDM/LongBench --repo-type dataset --local-dir /datasets/LongBench

# LongBench-v2. Ditto scripts read it from /datasets/LongBench-v2.
huggingface-cli download THUDM/LongBench-v2 --repo-type dataset --local-dir /datasets/LongBench-v2

# RULER generator and benchmark assets.
git clone https://github.com/NVIDIA/RULER.git /datasets/RULER
```

For RULER throughput runs, the generated JSONL files must be available under `test/ditto/speedup/data` with the names expected by the run scripts:

```shell
mkdir -p test/ditto/speedup/data

# Expected files after generating or copying RULER data:
#   test/ditto/speedup/data/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl
#   test/ditto/speedup/data/RULER-Qwen2.5-14B-Instruct-1M-16K.jsonl
#   ...
#   test/ditto/speedup/data/RULER-Qwen2.5-14B-Instruct-1M-512K.jsonl
#   test/ditto/speedup/data/RULER-Llama-3-8B-Instruct-Gradient-1048k-8K.jsonl
#   test/ditto/speedup/data/RULER-Llama-3-8B-Instruct-Gradient-1048k-16K.jsonl
#   ...
#   test/ditto/speedup/data/RULER-Llama-3-8B-Instruct-Gradient-1048k-512K.jsonl
```

## Preparations for Running

All test assets, configs, and benchmark scripts are under `test/ditto`.

The preparation pipeline is model-agnostic. Set the model path and the model-specific shape parameters from the target model config before training hash weights.

```shell
# From repo root
export MODEL_PATH=/models/Qwen2.5-14B-Instruct-1M
export MODEL_TAG="$(basename "${MODEL_PATH}")"
export LONGBENCH_PATH=/datasets/LongBench/data
export LONGBENCH_V2_PATH=/datasets/LongBench-v2
export LONGBENCH_ROOT=/datasets/LongBench

# Match these to the target model config.
export NUM_LAYERS=48
export NUM_HEADS=40
export NUM_KV_HEADS=8
export HEAD_DIM=128
export RBIT=256

cd test/ditto/auxiliary/hash_weights
python build_dataset.py \
  --model "${MODEL_PATH}" \
  --save_path "dataset/${MODEL_TAG}" \
  --longbench_path "${LONGBENCH_PATH}" \
  --longbench_v2_path "${LONGBENCH_V2_PATH}" \
  --max_context_length 65536 \
  --pos_sample_ratio 0.1 \
  --max_prompt_chars 0 \
  --pp_num 1 \
  --apply_template

python learn_hash_weights.py \
  --dataset_path "dataset/${MODEL_TAG}" \
  --save_path "${MODEL_TAG}-${RBIT}" \
  --num_layers "${NUM_LAYERS}" \
  --num_skip_layers 0 \
  --num_heads "${NUM_HEADS}" \
  --num_kv_heads "${NUM_KV_HEADS}" \
  --head_dim "${HEAD_DIM}" \
  --rbit "${RBIT}" \
  --chunk_num 3 \
  --mp_num 8 \
  --train_epochs 15 \
  --train_iters 20 \
  --rep_iters 10 \
  --lr 0.1 \
  --epsilon 0.01 \
  --lambdda 1.0 \
  --eta 2.0 \
  --sigma 0.1

cd ../attn_pattern
python profile_heads_cosine.py \
  --model "${MODEL_PATH}" \
  --dataset_path "${LONGBENCH_ROOT}" \
  --num_samples 10 \
  --max_context_length 65536 \
  --pp_num 1
```

The shell wrappers in these directories are preset examples; use the Python entrypoints above or copy a preset and adjust the defaults for a new model.

## Performance

### Offline Throughput

```shell
cd test/ditto/speedup

# Choose the target model, method, and sweep. For example:
bash run_n2n_qwen_offloading_seqlen.sh
bash run_n2n_qwen_offloading_bsz.sh
bash run_n2n_qwen_fullattn_seqlen.sh
bash run_n2n_qwen_fullattn_bsz.sh
```

For latency breakdown, try to run these scripts with `nsys profile`.

### Online Serving

Run the SGLang server in one terminal. Ditto offloading:

```shell
cd test/ditto/speedup
MODEL_PATH=/models/Qwen2.5-14B-Instruct-1M \
CUDA_DEVICE=0 \
PORT=30000 \
DITTO_MAX_BATCH_SIZE=32 \
DITTO_TARGET_SEQ_LEN=8192 \
bash ditto_server.sh
```

Dense / flash-attn baseline:

```shell
cd test/ditto/speedup
MODEL_PATH=/models/Qwen2.5-14B-Instruct-1M \
CUDA_DEVICE=0 \
PORT=30000 \
DITTO_MAX_BATCH_SIZE=16 \
DITTO_TARGET_SEQ_LEN=8192 \
bash flash_server.sh
```

The example `DITTO_MAX_BATCH_SIZE` values above are HBM-limited serving capacities, not fixed hyperparameters. Choose the largest max-request / max-batch value that fits the target model, sequence length, and GPU HBM budget; reduce it if the server OOMs. When `DITTO_TARGET_SEQ_LEN` is set, these scripts compute `DITTO_MAX_TOKENS = DITTO_TARGET_SEQ_LEN * DITTO_MAX_BATCH_SIZE`. If you also cap SGLang scheduling with `SGLANG_MAX_RUNNING_REQUESTS`, keep it no larger than the HBM-safe request count.

Then run the online RULER client sweep in another terminal:

```shell
cd test/ditto/speedup
python sweep_ruler_throughput.py \
  --model-path /models/Qwen2.5-14B-Instruct-1M \
  --server-base-url http://127.0.0.1:30000 \
  --data-file data/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl \
  --concurrencies 1,2,4,8,16 \
  --max-seq-len 8192 \
  --max-total-tokens 8192 \
  --max-new-tokens 128 \
  --output-dir results/online_serving_qwen_8k
```

`launch_client.py` is the per-point client used by the sweep. The `*_itl_tmp.sh` wrappers are ITL/debug helpers, not the main online-serving entrypoint.

### Speedup Ablation Study

Speedup ablation uses the pre-materialized profile configs under `test/ditto/speedup/tmp_config_B0` through `test/ditto/speedup/tmp_config_B6`. The wrapper scripts below set `CONFIG_ROOT` to the matching `tmp_config_B*` directory and do not rely on runtime `--ablation-profile` mutation.

All B0-B6 profiles keep HATA/hash retrieval and CPU KV-cache offloading enabled. The profiles cumulatively add transfer, prefetch, graph, similarity, adaptive threshold, and resident-cache optimizations:

| Profile | Transfer | Layer prefetch | Ditto CUDA graph | QSAC / similarity | Threshold | Resident cache |
| --- | --- | --- | --- | --- | --- | --- |
| `B0` | CUDA memcpy | off | off | off | fixed `1.0/1.0` | off |
| `B1` | GDRCopy | off | off | off | fixed `1.0/1.0` | off |
| `B2` | GDRCopy | on | off | off | fixed `1.0/1.0` | off |
| `B3` | GDRCopy | on | on | off | fixed `1.0/1.0` | off |
| `B4` | GDRCopy | on | on | on | fixed `0.8/0.8` | off |
| `B5` | GDRCopy | on | on | on | adaptive `-1.0/0.8` | off |
| `B6` | GDRCopy | on | on | on | adaptive `-1.0/0.8` | on |

In the current tmp configs, `num_skip_layers=1` for all profiles and SGLang global CUDA graph stays disabled; only Ditto CUDA graph changes from B3 onward.

```shell
cd test/ditto/speedup

# Run all B0-B6 tmp-config profiles in order.
MODEL_PATH=/models/Qwen2.5-14B-Instruct-1M \
PYTHON_BIN=python3 \
SEQ_LIST=8000 \
BSZ_LIST=64 \
bash run_n2n_qwen_offloading_bsz_B0_B6_sequence.sh

# Or run one profile, e.g. B4, which uses tmp_config_B4 by default.
MODEL_PATH=/models/Qwen2.5-14B-Instruct-1M \
SEQ_LIST=8000 \
BSZ_LIST=64 \
bash run_n2n_qwen_offloading_bsz_B4.sh
```

Equivalent direct command for a single `64x8K` B4 run; `64 * 8K = 512K`, so it uses the `512K` config in `tmp_config_B4`:

```shell
cd test/ditto/speedup
mkdir -p logs-batchless-B4
python n2n_offloading.py \
  --model /models/Qwen2.5-14B-Instruct-1M \
  --config_file tmp_config_B4/Qwen2.5-14B-Instruct-1M-512K.yaml \
  --data data/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl \
  --num_decode_steps 50 \
  --warmup 1 \
  --epoch 3 \
  --method offloading \
  --topk 0.10 \
  --batch_size 64 \
  --max_seq_len 8000 \
  --max-running-requests 64 \
  --mem-fraction-static 0.92 \
  --disable-cuda-graph \
  --ditto-enable-cuda-graph \
  --result-json logs-batchless-B4/B4-qwen2.5-14b-1m-offloading-bsz64-seq8K-topk0.10.json
```

Wrapper outputs are written under `logs-batchless-B0` through `logs-batchless-B6`.

## Accuracy

### RULER Dataset

RULER download and generated JSONL placement are described in `Data and Models`. The speedup scripts expect the generated RULER files under `test/ditto/speedup/data`.

### Main Results

```shell
cd test/ditto

# Choose the target benchmark. For example:
bash test_accuracy_math500.sh
bash test_accuracy_gpqa.sh
bash test_accuracy_aime24.sh
bash test_accuracy_aime25.sh
bash test_accuracy_infinitebench.sh
bash test_accuracy_mmlu_pro.sh
bash test_accuracy_multinews.sh
bash test_accuracy_livecodebench.sh
```

The accuracy wrappers call `test_accuracy_benchmark.sh`. Ditto offloading configs live under `test/ditto/config/hata_offloading`; dense full-attention baselines live under `test/ditto/config/full_attn`.
