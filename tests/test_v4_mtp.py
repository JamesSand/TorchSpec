# Copyright (c) 2026 LightSeek Foundation
#
# Tests for the DeepSeek-V4 native MTP/NextN draft (deepseek_v4 arch) ported into
# TorchSpec: single-GPU forward/backward + Eagle3 multi-step TTT training path, and
# Expert-Parallel (spawned) parity. See torchspec/models/draft/deepseek_v4_nextn_eagle.py.

import json
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DT = torch.bfloat16
_CFG = "/data/zhizhousha/dpsk/DeepSeek-V4-Flash"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tiny_config():
    """A small, self-consistent deepseek_v4 config derived from DeepSeek-V4-Flash."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    real = json.load(open(os.path.join(_CFG, "config.json")))
    over = dict(
        num_hidden_layers=1, hidden_size=256, num_attention_heads=4, head_dim=64,
        num_key_value_heads=1, q_lora_rank=64, o_lora_rank=64, o_groups=2,
        intermediate_size=128, moe_intermediate_size=128, n_routed_experts=8,
        num_local_experts=8, num_experts_per_tok=2, n_shared_experts=1, vocab_size=512,
        index_head_dim=32, index_n_heads=4, index_topk=16, hc_mult=4, hc_sinkhorn_iters=4,
        compress_ratios=[0], layer_types=["sliding_attention"], mlp_layer_types=["moe"],
        num_hash_layers=0, sliding_window=128, max_position_embeddings=2048,
        num_aux_hidden_states=1,
    )
    return DeepseekV4Config(**{**real, **over})


@unittest.skipUnless(os.path.isdir(_CFG), "requires the DeepSeek-V4-Flash config")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestV4MTPSingleGPU(unittest.TestCase):
    def test_forward_backward(self):
        from torchspec.models.draft.deepseek_v4_nextn_eagle import Eagle3DeepseekV4ForCausalLM

        dev = "cuda"
        cfg = tiny_config()
        m = Eagle3DeepseekV4ForCausalLM(cfg, attention_backend="eager").to(dev, DT)
        B, S, H, V = 2, 16, cfg.hidden_size, cfg.vocab_size
        emb = m.embed_input_ids(torch.randint(0, V, (B, S), device=dev))
        hid = torch.randn(B, S, H, device=dev, dtype=DT, requires_grad=True)
        pos = torch.arange(S, device=dev).unsqueeze(0)
        mask = m.prepare_decoder_attention_mask(torch.ones(B, S, device=dev, dtype=torch.long),
                                                hid, B, S, 0)
        out, _, _ = m.backbone(emb, hid, attention_mask=mask, position_ids=pos)
        self.assertEqual(out.shape, (B, S, H))
        self.assertTrue(torch.isfinite(out).all())
        m.compute_logits(out).float().pow(2).mean().backward()
        self.assertIsNotNone(m.midlayer.mlp.experts.gate_up_proj.grad)
        self.assertIsNotNone(hid.grad)

    def test_eagle3_ttt(self):
        from torchspec import AutoEagle3DraftModel
        from torchspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded
        from torchspec.training.optimizer import BF16Optimizer

        dev = "cuda"
        cfg = tiny_config()
        draft = AutoEagle3DraftModel.from_config(cfg, attention_backend="eager", torch_dtype=DT).to(dev)
        self.assertEqual(type(draft).__name__, "Eagle3DeepseekV4ForCausalLM")
        model = Eagle3Model(draft_model=draft, length=4, attention_backend="eager",
                            gradient_checkpointing=False).to(dev)
        opt = BF16Optimizer(draft, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0)
        B, S, H, V = 2, 12, cfg.hidden_size, cfg.vocab_size
        ids = torch.randint(0, V, (B, S), device=dev)
        target = compute_lazy_target_padded(torch.randn(B, S, H, device=dev, dtype=DT),
                                            torch.randn(V, H, device=dev, dtype=DT), 4)
        before = draft.midlayer.mlp.experts.gate_up_proj.detach().clone()
        losses = []
        for _ in range(3):
            opt.zero_grad()
            plosses, *_ = model(input_ids=ids,
                                attention_mask=torch.ones(B, S, device=dev, dtype=torch.long),
                                target=target, loss_mask=torch.ones(B, S, device=dev),
                                hidden_states=torch.randn(B, S, H, device=dev, dtype=DT),
                                position_ids=None)
            loss = sum(plosses) / len(plosses)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in losses))
        self.assertFalse(torch.equal(before, draft.midlayer.mlp.experts.gate_up_proj.detach()))


def _ep_worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    ep_group = dist.group.WORLD

    from torchspec import AutoEagle3DraftModel
    from torchspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded
    from torchspec.training.ep_utils import sync_gradients_ep
    from torchspec.training.optimizer import BF16Optimizer

    cfg = tiny_config()
    cfg.ep_group = ep_group
    draft = AutoEagle3DraftModel.from_config(cfg, attention_backend="eager", torch_dtype=DT).to(dev)
    with torch.no_grad():
        for _n, p in draft.named_parameters():
            if not getattr(p, "_is_ep", False):
                dist.broadcast(p.data, src=0)
    model = Eagle3Model(draft_model=draft, length=4, attention_backend="eager",
                        gradient_checkpointing=False).to(dev)
    opt = BF16Optimizer(draft, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0,
                        ep_group=ep_group)
    assert opt.has_ep and sum(opt.ep_mask) > 0

    B, S, H, V = 2, 12, cfg.hidden_size, cfg.vocab_size
    torch.manual_seed(0)
    ids = torch.randint(0, V, (B, S), device=dev)
    target = compute_lazy_target_padded(torch.randn(B, S, H, device=dev, dtype=DT),
                                        torch.randn(V, H, device=dev, dtype=DT), 4)
    before = draft.midlayer.mlp.experts.gate_up_proj.detach().clone()
    gn = None
    for _ in range(3):
        opt.zero_grad()
        plosses, *_ = model(input_ids=ids,
                            attention_mask=torch.ones(B, S, device=dev, dtype=torch.long),
                            target=target, loss_mask=torch.ones(B, S, device=dev),
                            hidden_states=torch.randn(B, S, H, device=dev, dtype=DT), position_ids=None)
        loss = sum(plosses) / len(plosses)
        loss.backward()
        sync_gradients_ep(opt.model_params, dp_group=None, ep_fsdp_group=None)
        gn = opt.step().item()
    assert not torch.equal(before, draft.midlayer.mlp.experts.gate_up_proj.detach()), "experts not updated"
    g = torch.tensor([gn], device=dev)
    gathered = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(gathered, g)
    assert all(abs(t.item() - gn) < 1e-2 for t in gathered), "EP grad-norm inconsistent"
    dist.destroy_process_group()


@unittest.skipUnless(os.path.isdir(_CFG), "requires the DeepSeek-V4-Flash config")
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires >= 2 GPUs")
class TestV4MTPExpertParallel(unittest.TestCase):
    def test_ep_training_step(self):
        mp.spawn(_ep_worker, args=(2, _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
