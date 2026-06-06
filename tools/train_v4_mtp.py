"""Train the DeepSeek-V4 native MTP speculator (Eagle3 multi-step TTT) from the
MTP-initialized checkpoint, on real collected (input_ids, last_hidden) data.
Logs simulated_acc_len to W&B. Supports Expert Parallel via torchrun.

Single GPU:
  python tools/train_v4_mtp.py --data .../v4_collected_train.pt --eval-data .../v4_collected_eval.pt
Expert Parallel (ep_size == world_size):
  torchrun --standalone --nproc_per_node=8 tools/train_v4_mtp.py --data ... --eval-data ... --wandb online

simulated_acc_len = acc_0 + acc_0*acc_1 + ... (matches eagle3_trainer aggregation).
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

CKPT = "/data/zhizhousha/dpsk/DeepSeek-V4-Flash"


def simulated_acc_len(acces):
    cum, total = 1.0, 0.0
    for a in acces:
        cum *= float(a)
        total += cum
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="/data/zhizhousha/dpsk/v4flash_mtp_init")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ttt", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss", default="kl", choices=["kl", "ce"])
    ap.add_argument("--wandb", default="offline", choices=["offline", "online", "off"])
    ap.add_argument("--run-name", default="v4mtp")
    args = ap.parse_args()

    DT = torch.bfloat16
    ep = "RANK" in os.environ
    if ep:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        ep_group = dist.group.WORLD
    else:
        rank, world, dev, ep_group = 0, 1, torch.device("cuda"), None
    is_main = rank == 0

    def log0(*a):
        if is_main:
            print(*a, flush=True)

    import torch.nn.functional as F
    from transformers import AutoConfig

    from torchspec import AutoEagle3DraftModel
    from torchspec.models.eagle3 import (
        Eagle3Model,
        PrecomputedTarget,
        compute_lazy_target_padded,
    )
    from torchspec.training.ep_utils import sync_gradients_ep
    from torchspec.training.optimizer import BF16Optimizer

    cfg = AutoConfig.from_pretrained(args.init, trust_remote_code=True)
    cfg.num_aux_hidden_states = 1
    if ep_group is not None:
        cfg.ep_group = ep_group
    draft = AutoEagle3DraftModel.from_config(cfg, attention_backend="eager", torch_dtype=DT).to(dev)

    # --- load MTP-init: shard expert params per rank (EP), load the rest directly ---
    full_sd = torch.load(os.path.join(args.init, "mtp_init_state.pt"), map_location="cpu")
    own = draft.state_dict()
    load_sd = {}
    for k, v in full_sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            load_sd[k] = v
        elif "experts" in k and v.shape[0] % own[k].shape[0] == 0:
            nl = own[k].shape[0]  # local expert count
            load_sd[k] = v[rank * nl:(rank + 1) * nl].clone()
        # else: shape mismatch -> leave to model init (shouldn't happen)
    incompat = draft.load_state_dict(load_sd, strict=False)
    # belt-and-suspenders: broadcast non-expert (replicated) params from rank 0
    if ep_group is not None:
        with torch.no_grad():
            for _n, p in draft.named_parameters():
                if not getattr(p, "_is_ep", False):
                    dist.broadcast(p.data, src=0)
    draft.freeze_embedding()
    log0(f"[init] EP={ep} world={world} loaded MTP-init "
         f"(missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)})")

    model = Eagle3Model(draft_model=draft, length=args.ttt, attention_backend="eager",
                        gradient_checkpointing=True).to(dev)
    opt = BF16Optimizer(draft, lr=args.lr, max_grad_norm=1.0, total_steps=args.steps,
                        warmup_ratio=0.05, ep_group=ep_group)
    log0(f"[opt] has_ep={opt.has_ep} ep_params={sum(opt.ep_mask)}/{len(opt.ep_mask)}")

    H = cfg.hidden_size

    # --- W&B (rank 0 only) ---
    run = None
    if is_main and args.wandb != "off":
        import re

        import wandb
        if args.wandb == "online":
            kf = "/data/zhizhousha/workspace/aurora-project/api.txt"
            key = None
            if os.path.exists(kf):
                for line in open(kf):
                    s = line.strip()
                    if s.startswith("wandb_v1_") or re.fullmatch(r"[A-Za-z0-9]{40}", s):
                        key = s
                        break
            if key:
                os.environ["WANDB_API_KEY"] = key
            else:
                os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "offline"
        try:
            run = wandb.init(project="dpsk-v4-mtp-eagle3", name=args.run_name,
                             dir="/data/zhizhousha/dpsk", config=vars(args))
        except Exception as e:
            print(f"[wandb] init failed ({e}); offline", flush=True)
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(project="dpsk-v4-mtp-eagle3", name=args.run_name,
                             dir="/data/zhizhousha/dpsk", config=vars(args), reinit=True)

    # --- target lm_head (head.weight, bf16) for the distillation target ---
    from safetensors import safe_open
    widx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(CKPT, widx["head.weight"]), framework="pt") as f:
        head_weight = f.get_tensor("head.weight").to(dev, DT)

    hc_mult = cfg.hc_mult
    V = head_weight.shape[0]

    def make_target(lh):
        """KL: soft target_p (distillation). CE: one-hot of target's argmax token."""
        if args.loss == "kl":
            return compute_lazy_target_padded(lh, head_weight, args.ttt)
        Bb, Tt, _ = lh.shape
        gt = torch.matmul(lh.float(), head_weight.float().t()).argmax(-1)  # [B,T]
        onehot = F.one_hot(gt, V).to(torch.float32)
        tp = F.pad(onehot, (0, 0, 0, args.ttt), value=0.0)
        return PrecomputedTarget(tp, torch.ones(Bb, Tt, device=lh.device))

    def prep(path):
        raw = torch.load(path, map_location="cpu")
        out = []
        with torch.no_grad():
            for s in raw["samples"]:
                h = s["hidden"]
                Tp = h.shape[0] // hc_mult
                if Tp < 2:
                    continue
                streams = h[: Tp * hc_mult].reshape(1, Tp, hc_mult, H).to(dev, DT)
                last_hidden = model.draft_model.norm(model.draft_model.hc_head(streams))
                ids = s["input_ids"][:Tp].to(dev)[None]
                out.append((ids, last_hidden.detach()))
        return out

    train_set = prep(args.data)
    eval_set = prep(args.eval_data) if args.eval_data else train_set[: min(32, len(train_set))]
    log0(f"[data] train={len(train_set)} eval={len(eval_set)} loss={args.loss}")

    def run_eval():
        model.eval()
        vals = []
        with torch.no_grad():
            for ids, lh in eval_set:
                Bb, Sb = ids.shape
                tgt = make_target(lh)
                _, _, acces, _ = model(input_ids=ids,
                                       attention_mask=torch.ones(Bb, Sb, device=dev, dtype=torch.long),
                                       target=tgt, loss_mask=torch.ones(Bb, Sb, device=dev),
                                       hidden_states=lh, position_ids=None)
                vals.append(simulated_acc_len([float(a) for a in acces]))
        model.train()
        return sum(vals) / len(vals)

    base = run_eval()
    log0(f"[BASELINE] MTP-init eval/simulated_acc_len = {base:.3f}")
    if run:
        run.log({"eval/simulated_acc_len": base}, step=0)

    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(args.steps):
        ids, lh = train_set[step % len(train_set)]
        B, S = ids.shape
        opt.zero_grad()
        target = make_target(lh)
        plosses, _, acces, _ = model(input_ids=ids,
                                     attention_mask=torch.ones(B, S, device=dev, dtype=torch.long),
                                     target=target, loss_mask=torch.ones(B, S, device=dev),
                                     hidden_states=lh, position_ids=None)
        loss = sum(plosses) / len(plosses)
        loss.backward()
        if ep_group is not None:
            sync_gradients_ep(opt.model_params, dp_group=None, ep_fsdp_group=None)
        gn = opt.step().item()
        if is_main and (step % 20 == 0 or step == args.steps - 1):
            sal = simulated_acc_len([float(a) for a in acces])
            log0(f"[step {step}] loss={loss.item():.4f} gn={gn:.3f} train_sal={sal:.3f}")
        if run and step % 20 == 0:
            run.log({"train/loss": loss.item(), "train/grad_norm": gn}, step=step)
        if (step + 1) % 100 == 0:
            ev = run_eval()
            log0(f"[eval @ step {step+1}] simulated_acc_len = {ev:.3f}")
            if run:
                run.log({"eval/simulated_acc_len": ev}, step=step + 1)

    final = run_eval()
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    log0(f"[DONE] baseline={base:.3f} -> final eval/simulated_acc_len={final:.3f} "
         f"(loss={args.loss}, peak_mem/rank={peak:.1f}GB)")
    if run:
        run.log({"eval/simulated_acc_len": final}, step=args.steps)
        run.finish()
    if ep_group is not None:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
