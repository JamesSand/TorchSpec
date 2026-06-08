"""MTP-init: build the TorchSpec DeepSeek-V4 MTP draft and initialize it from the
target's native `mtp.0.*` weights (DeepSeek-V4-Flash native checkpoint).

Handles the mixed quantization in the native checkpoint:
  - BF16 / F32 (no .scale): norms, sinks, gate, hyper-connection, embed, head -> direct.
  - FP8 e4m3 + e8m0 block-128 scale: attention projs, e_proj/h_proj, shared experts.
  - MXFP4 (fp4 e2m1, 2/byte) + e8m0 block-32 scale: routed experts w1/w2/w3.

Native key (mtp.0.*) -> transformers-style draft param mapping (see KEY_MAP).

Usage:
  OMP_PROC_BIND=FALSE KMP_AFFINITY=disabled taskset -c 0-15 \
    python tools/init_draft_from_mtp.py \
      --target /data/zhizhousha/dpsk/DeepSeek-V4-Flash \
      --out-dir /data/zhizhousha/dpsk/v4flash_mtp_init
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

# fp4 e2m1 lookup table (index = 4-bit code): magnitudes {0,.5,1,1.5,2,3,4,6}, sign bit high.
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _block_expand(scale: torch.Tensor, out_rows: int, out_cols: int) -> torch.Tensor:
    s = scale.float()
    return s.repeat_interleave(out_rows // s.shape[0], 0).repeat_interleave(out_cols // s.shape[1], 1)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 block-128 dequant (e8m0 scale). weight [nr,nc], scale [nr/128, nc/128]."""
    nr, nc = weight.shape
    w = weight.float()
    se = _block_expand(scale, nr, nc)
    return (w * se).to(torch.bfloat16)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 (2 fp4 e2m1 / byte) + e8m0 block-32 scale. packed int8 [O, Ip] -> bf16 [O, 2*Ip]."""
    b = packed.contiguous().view(torch.uint8).to(torch.int64)
    lut = _E2M1.to(b.device)
    lo = lut[b & 0xF]
    hi = lut[(b >> 4) & 0xF]
    nr, n_packed = packed.shape
    out = torch.empty(nr, n_packed * 2, dtype=torch.float32, device=b.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    se = _block_expand(scale, nr, n_packed * 2)
    return (out * se).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/data/zhizhousha/dpsk/DeepSeek-V4-Flash")
    ap.add_argument("--out-dir", default="/data/zhizhousha/dpsk/v4flash_mtp_init")
    args = ap.parse_args()

    from transformers import AutoConfig

    from torchspec.models.draft.deepseek_v4_nextn_eagle import Eagle3DeepseekV4ForCausalLM

    d = args.target
    index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    # cache open handles per shard
    _handles = {}

    def _h(fn):
        if fn not in _handles:
            _handles[fn] = safe_open(os.path.join(d, fn), framework="pt")
        return _handles[fn]

    def get(k):
        return _h(index[k]).get_tensor(k) if k in index else None

    def has(k):
        return k in index

    # --- build draft (CPU bf16) ---
    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    cfg.num_aux_hidden_states = 1
    cfg.mlp_layer_types = list(cfg.mlp_layer_types)
    cfg.mlp_layer_types[0] = "moe"  # MTP gate has weight+bias -> TopK router (+correction bias)
    model = Eagle3DeepseekV4ForCausalLM(cfg, attention_backend="eager").to(torch.bfloat16)
    sd = model.state_dict()

    E = cfg.n_routed_experts
    used = set()

    def load_w(native_key):
        """Load a (possibly quantized) native weight -> bf16 tensor."""
        used.add(native_key)
        w = get(native_key)
        sk = native_key.replace(".weight", ".scale")
        if native_key.endswith(".weight") and has(sk):
            used.add(sk)
            s = get(sk)
            if w.dtype == torch.float8_e4m3fn:
                return dequant_fp8(w, s)
            if w.dtype == torch.int8:
                return dequant_mxfp4(w, s)
            raise ValueError(f"{native_key}: unexpected quant dtype {w.dtype}")
        return w.to(torch.bfloat16) if w.dtype != torch.float32 else w

    # --- direct / fp8 mappings (draft param name -> native key) ---
    M = {
        "embed_tokens.weight": "embed.weight",
        "lm_head.weight": "head.weight",
        "norm.weight": "mtp.0.norm.weight",
        "enorm.weight": "mtp.0.enorm.weight",
        "hnorm.weight": "mtp.0.hnorm.weight",
        "e_proj.weight": "mtp.0.e_proj.weight",
        "h_proj.weight": "mtp.0.h_proj.weight",
        "midlayer.self_attn.sinks": "mtp.0.attn.attn_sink",
        "midlayer.self_attn.q_a_proj.weight": "mtp.0.attn.wq_a.weight",
        "midlayer.self_attn.q_a_norm.weight": "mtp.0.attn.q_norm.weight",
        "midlayer.self_attn.q_b_proj.weight": "mtp.0.attn.wq_b.weight",
        "midlayer.self_attn.kv_proj.weight": "mtp.0.attn.wkv.weight",
        "midlayer.self_attn.kv_norm.weight": "mtp.0.attn.kv_norm.weight",
        "midlayer.self_attn.o_a_proj.weight": "mtp.0.attn.wo_a.weight",
        "midlayer.self_attn.o_b_proj.weight": "mtp.0.attn.wo_b.weight",
        "midlayer.input_layernorm.weight": "mtp.0.attn_norm.weight",
        "midlayer.post_attention_layernorm.weight": "mtp.0.ffn_norm.weight",
        "midlayer.mlp.gate.weight": "mtp.0.ffn.gate.weight",
        "midlayer.mlp.shared_experts.gate_proj.weight": "mtp.0.ffn.shared_experts.w1.weight",
        "midlayer.mlp.shared_experts.up_proj.weight": "mtp.0.ffn.shared_experts.w3.weight",
        "midlayer.mlp.shared_experts.down_proj.weight": "mtp.0.ffn.shared_experts.w2.weight",
        "midlayer.attn_hc.fn": "mtp.0.hc_attn_fn",
        "midlayer.attn_hc.base": "mtp.0.hc_attn_base",
        "midlayer.attn_hc.scale": "mtp.0.hc_attn_scale",
        "midlayer.ffn_hc.fn": "mtp.0.hc_ffn_fn",
        "midlayer.ffn_hc.base": "mtp.0.hc_ffn_base",
        "midlayer.ffn_hc.scale": "mtp.0.hc_ffn_scale",
        "hc_head.hc_fn": "mtp.0.hc_head_fn",
        "hc_head.hc_base": "mtp.0.hc_head_base",
        "hc_head.hc_scale": "mtp.0.hc_head_scale",
    }
    new_sd = {}
    for dst, src in M.items():
        t = load_w(src)
        assert tuple(t.shape) == tuple(sd[dst].shape), f"{dst}: {tuple(t.shape)} != {tuple(sd[dst].shape)}"
        new_sd[dst] = t

    # router correction bias buffer <- native gate.bias
    new_sd["midlayer.mlp.gate.e_score_correction_bias"] = get("mtp.0.ffn.gate.bias").float()
    used.add("mtp.0.ffn.gate.bias")

    # --- routed experts: stack per-expert w1/w3 -> gate_up_proj (E,2I,H); w2 -> down_proj (E,H,I) ---
    # TorchSpec MoEExperts layout: gate_up_proj [E, H, 2I], down_proj [E, I, H]
    # (grouped_mm expects w[E, K=in, N=out]). Native w1/w3 are [I,H], w2 is [H,I] -> transpose.
    gate_up, down = [], []
    for e in range(E):
        w1 = load_w(f"mtp.0.ffn.experts.{e}.w1.weight")  # gate [I,H]
        w3 = load_w(f"mtp.0.ffn.experts.{e}.w3.weight")  # up   [I,H]
        w2 = load_w(f"mtp.0.ffn.experts.{e}.w2.weight")  # down [H,I]
        gate_up.append(torch.cat([w1, w3], dim=0).t().contiguous())  # [2I,H] -> [H,2I]
        down.append(w2.t().contiguous())                             # [H,I] -> [I,H]
    new_sd["midlayer.mlp.experts.gate_up_proj"] = torch.stack(gate_up, 0)
    new_sd["midlayer.mlp.experts.down_proj"] = torch.stack(down, 0)
    for nm in ["midlayer.mlp.experts.gate_up_proj", "midlayer.mlp.experts.down_proj"]:
        assert tuple(new_sd[nm].shape) == tuple(sd[nm].shape), f"{nm} shape mismatch"

    # rotary inv_freq buffers: keep the freshly-computed ones from model init
    for k in sd:
        if "rotary_emb" in k or k.endswith(".t2d") or k.endswith(".d2t"):
            new_sd[k] = sd[k]

    missing = [k for k in sd if k not in new_sd]
    print(f"[map] loaded {len(new_sd)}/{len(sd)} draft tensors; missing={missing}", flush=True)
    incompat = model.load_state_dict(new_sd, strict=False)
    print(f"[load] missing_keys={list(incompat.missing_keys)}", flush=True)
    print(f"[load] unexpected_keys={list(incompat.unexpected_keys)}", flush=True)

    # --- forward sanity on the real-dim draft (CPU, tiny batch) ---
    model.eval()
    B, S, H, V = 1, 8, cfg.hidden_size, cfg.vocab_size
    ids = torch.randint(0, V, (B, S))
    emb = model.embed_input_ids(ids)
    hid = torch.randn(B, S, H, dtype=torch.bfloat16)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        out, _, _ = model.backbone(emb, hid, attention_mask=torch.ones(B, S, dtype=torch.long), position_ids=pos)
        logits = model.compute_logits(out)
    print(f"[sanity] backbone finite={torch.isfinite(out).all().item()} "
          f"logits finite={torch.isfinite(logits).all().item()} "
          f"logits std={logits.float().std().item():.3f}", flush=True)

    # report which native mtp keys were NOT consumed (e.g. dropped indexer/compressor — should be none for MTP)
    all_mtp = [k for k in index if k.startswith("mtp.0.")]
    unused = sorted(set(all_mtp) - used)
    print(f"[coverage] native mtp keys={len(all_mtp)} used={len(used & set(all_mtp))} unused={len(unused)}", flush=True)
    if unused:
        print("  unused sample:", unused[:10], flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "mtp_init_state.pt"))
    cfg.save_pretrained(args.out_dir)
    print(f"[OK] MTP-init draft saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
