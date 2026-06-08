"""Collect (input_ids, last_hidden) training data from DeepSeek-V4-Flash served by
sglang (tgl fork). Loads the engine ONCE and collects both train + eval splits.

Run with the SGLANG venv (B200 LowLatency recipe; flash_mla built for sm_100):
  OMP_PROC_BIND=FALSE KMP_AFFINITY=disabled taskset -c 0-15 \
    /data/zhizhousha/sgl_venv/bin/python tools/collect_v4_data.py --tp 4 \
      --num-train 512 --num-eval 64 --max-new 256

Output .pt per split: {"samples":[{"hidden":[T*hc_mult,H], "input_ids":[T], "text":..}],
                       "model":.., "hidden_size":H}
last_hidden = hc_head-collapse + norm of the streams (done train-side). DO NOT hard-kill
mid-load (wedges GPUs); let it finish/exit cleanly.
"""

import argparse
import json

import torch

TRAIN_FILE = "examples/data/sample_conversations.jsonl"
EVAL_FILE = "examples/data/eval_conversations.jsonl"


def load_prompts(path, n):
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            convs = row.get("conversations", [])
            user = next((c["content"] for c in convs if c.get("role") == "user"), None)
            if isinstance(user, str) and user.strip():
                prompts.append(user.strip())
            if len(prompts) >= n:
                break
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhizhousha/dpsk/DeepSeek-V4-Flash")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--num-train", type=int, default=512)
    ap.add_argument("--num-eval", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out-train", default="/data/zhizhousha/dpsk/v4_collected_train.pt")
    ap.add_argument("--out-eval", default="/data/zhizhousha/dpsk/v4_collected_eval.pt")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoConfig, AutoTokenizer

    H = AutoConfig.from_pretrained(args.model, trust_remote_code=True).hidden_size
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def parse_hidden(hs, dbg=False):
        import numpy as np
        parts = [torch.as_tensor(np.asarray(e, dtype=np.float32)).reshape(-1) for e in hs]
        flat = torch.cat(parts)
        if dbg:
            print(f"[debug] hs list len={len(hs)} elem0={parts[0].numel()} total={flat.numel()} "
                  f"/H={flat.numel()/H:.2f}", flush=True)
        if flat.numel() % H == 0:
            return flat.reshape(-1, H)
        if flat.numel() % (4 * H) == 0:
            return flat.reshape(-1, 4 * H)
        raise ValueError(f"hidden total {flat.numel()} not divisible by H={H} or 4H")

    print(f"[engine] launching V4-Flash tp={args.tp} (B200 LowLatency recipe)...", flush=True)
    engine = sgl.Engine(
        model_path=args.model, tp_size=args.tp, trust_remote_code=True,
        mem_fraction_static=args.mem_frac, moe_runner_backend="flashinfer_mxfp4",
        chunked_prefill_size=4096, disable_flashinfer_autotune=True,
        enable_return_hidden_states=True, log_level="info",
    )
    print("[engine] ready.", flush=True)

    sampling = {"temperature": 0.7, "max_new_tokens": args.max_new, "top_p": 0.95}

    def collect(prompts, out_path, tag):
        samples = []
        for i, p in enumerate(prompts):
            try:
                out = engine.generate(p, sampling, return_hidden_states=True)
            except Exception as e:
                print(f"[{tag}] prompt {i} generate failed: {repr(e)[:120]}", flush=True)
                continue
            mi = out.get("meta_info", {})
            hs = mi.get("hidden_states") or out.get("hidden_states")
            if not hs:
                continue
            h = parse_hidden(hs, dbg=(i == 0 and tag == "train")).cpu()
            prompt_ids = tok.encode(p)
            out_ids = out.get("output_ids") or mi.get("output_token_ids") or []
            input_ids = torch.tensor(list(prompt_ids) + list(out_ids), dtype=torch.long)
            samples.append({"hidden": h, "input_ids": input_ids, "text": p})
            if i % 32 == 0:
                print(f"[{tag}] {i+1}/{len(prompts)} hidden={tuple(h.shape)} ids={input_ids.shape[0]}", flush=True)
        torch.save({"samples": samples, "model": args.model, "hidden_size": H}, out_path)
        print(f"[OK] {tag}: saved {len(samples)} samples -> {out_path}", flush=True)

    train_prompts = load_prompts(TRAIN_FILE, args.num_train)
    eval_prompts = load_prompts(EVAL_FILE, args.num_eval)
    print(f"[prompts] train={len(train_prompts)} eval={len(eval_prompts)}", flush=True)
    collect(train_prompts, args.out_train, "train")
    collect(eval_prompts, args.out_eval, "eval")
    engine.shutdown()


if __name__ == "__main__":
    main()
