# Training the DeepSeek-V4-Flash native MTP speculator (TorchSpec)

Train DeepSeek-V4-Flash's **native MTP/NextN module** as an Eagle3-style draft
(speculator) in TorchSpec: warm-start from the shipped `mtp.0.*` weights, collect
the target's hidden states, then run multi-step TTT training. Expert Parallel (EP)
across 8 GPUs.

**Result (8×B200, held-out eval):** `simulated_acc_len` 0.74 (MTP-init) → **1.66**
(KL distill) / 1.635 (CE). W&B project `dpsk-v4-mtp-eagle3`.

---

## 0. What's where

| Thing | Path |
|-------|------|
| Target model (quantized FP4+FP8) | `/data/zhizhousha/dpsk/DeepSeek-V4-Flash` (HF `deepseek-ai/DeepSeek-V4-Flash`, ~149G) |
| Training venv (torch 2.9.1+cu128) | `TorchSpec/.venv` |
| sglang venv (tgl fork + sm_100 flash_mla, torch 2.11) | `/data/zhizhousha/sgl_venv` |
| MTP-init checkpoint (output of step 1) | `/data/zhizhousha/dpsk/v4flash_mtp_init/` |
| Collected data (output of step 2) | `/data/zhizhousha/dpsk/v4_collected_{train,eval}.pt` |
| W&B key | `/data/zhizhousha/workspace/aurora-project/api.txt` (train script auto-reads) |

**Code (all in this repo):**
`torchspec/models/draft/deepseek_v4_nextn_eagle.py` (the draft) ·
`tools/init_draft_from_mtp.py` · `tools/collect_v4_data.py` · `tools/train_v4_mtp.py` ·
`tests/test_v4_mtp.py`. EP infra reused: `torchspec/models/draft/{moe,moe_ep}.py`,
`torchspec/training/{ep_utils,optimizer}.py`.

> ⚠️ **This box has a wedged CPU core** — every torch command MUST be prefixed to pin
> to good cores, or `import torch` hangs:
> ```bash
> ENV="OMP_PROC_BIND=FALSE KMP_AFFINITY=disabled OMP_NUM_THREADS=8 taskset -c 0-15"
> ```
> Also: never hard-kill a multi-GPU job mid-launch (wedges the GPUs).

The MTP-init checkpoint and collected data already exist on disk, so **to just re-train
you only need Step 3.**

---

## 1. MTP-init — build the draft from the target's native MTP weights (once, CPU)

Reads `mtp.0.*` from the quantized checkpoint (FP8 e4m3 + MXFP4 + bf16), dequantizes
everything to bf16, maps it into the TorchSpec draft, saves a loadable checkpoint.

```bash
cd TorchSpec && source .venv/bin/activate
$ENV python tools/init_draft_from_mtp.py \
    --target /data/zhizhousha/dpsk/DeepSeek-V4-Flash \
    --out-dir /data/zhizhousha/dpsk/v4flash_mtp_init
```
Expect: `loaded 33/33 draft tensors; missing=[]`, `native mtp keys=1575 used=1575 unused=0`,
`backbone finite=True`.

## 2. Collect target hidden states (sglang venv, 4 GPUs)

Serves the quantized V4-Flash via sglang (tgl fork; B200 LowLatency recipe:
`flashinfer_mxfp4` MoE + the self-built sm_100 `flash_mla`, no DeepEP),
runs prompts with `return_hidden_states=True`, saves `(input_ids, last_hidden streams)`.

```bash
$ENV CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /data/zhizhousha/sgl_venv/bin/python tools/collect_v4_data.py \
    --tp 4 --num-train 512 --num-eval 64 --max-new 256
# -> /data/zhizhousha/dpsk/v4_collected_train.pt  (512 samples)
#    /data/zhizhousha/dpsk/v4_collected_eval.pt   (64 samples)
```
First launch warms up DeepGEMM/cuda-graphs (~10–15 min); then it generates. Let it
finish cleanly (do not Ctrl-C mid-load).

## 3. Train the MTP draft — Expert Parallel, 8 GPUs

Multi-step TTT. `--loss kl` = distill the target distribution (recommended);
`--loss ce` = cross-entropy to the target's top-1 token.

```bash
cd TorchSpec && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$PWD $ENV \
  torchrun --standalone --nproc_per_node=8 tools/train_v4_mtp.py \
    --data      /data/zhizhousha/dpsk/v4_collected_train.pt \
    --eval-data /data/zhizhousha/dpsk/v4_collected_eval.pt \
    --steps 600 --ttt 4 --lr 1e-4 \
    --loss kl --wandb online --run-name v4mtp-KL-ep8
# CE run: --loss ce --run-name v4mtp-CE-ep8
```
It prints `[BASELINE] MTP-init eval/simulated_acc_len = …`, evals every 100 steps,
ends with `[DONE] baseline=… -> final eval/simulated_acc_len=…`. ~40 GB/GPU.

**Single-GPU fallback** (collected seqs are short, fits one B200; just drop torchrun):
```bash
CUDA_VISIBLE_DEVICES=0 $ENV python tools/train_v4_mtp.py \
    --data ..._train.pt --eval-data ..._eval.pt --loss kl --wandb offline
```

## 4. Tests

```bash
cd TorchSpec && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD $ENV python -m pytest tests/test_v4_mtp.py -v
# single-GPU fwd/bwd + Eagle3 TTT + (spawned) EP — needs >=2 GPUs for the EP test
```

---

## How it fits together
```
quantized V4-Flash (sglang, FP4/FP8)  --forward-->  last_hidden + target dist   [step 2]
                                                            │ (distillation target)
mtp.0.* (FP8/FP4) --dequant--> bf16 MTP draft  --multi-step TTT (EP)-->  raises acc length   [steps 1,3]
```
- Only the **MTP draft** is dequantized to bf16 and trained; the **target stays quantized**
  and is only used to produce hidden states.
- `simulated_acc_len = acc_0 + acc_0·acc_1 + …` = expected accepted draft tokens
  (conventional acceptance length = 1 + this).

Details: notes 006 (results), 007 (per-module quantization), 008 (structure + data format).
