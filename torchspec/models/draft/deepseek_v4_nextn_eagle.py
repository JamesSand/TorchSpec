# Copyright (c) 2026 LightSeek Foundation
#
# Eagle3 / MTP draft model for DeepSeek-V4 (deepseek_v4 arch), ported from the
# native NextN module (sglang `deepseek_v4_nextn.py`) into a training-friendly
# TorchSpec draft. Reuses transformers `modeling_deepseek_v4` building blocks
# (DecoderLayer = MQA sliding-window attn + mHC hyper-connection + Sparse MoE).
#
# The MTP/NextN layer is layer_idx=0 -> layer_type "sliding_attention": it has NO
# DSA compressor / Lightning Indexer (those live only in the CSA/HCA middle layers
# of the full model). So this draft layer is a plain sliding-window MQA layer +
# manifold-constrained hyper-connections (Sinkhorn, differentiable) + (hash) MoE.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4MLP,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4TopKRouter,
)

from torchspec.models.draft.base import Eagle3DraftModel
from torchspec.models.draft.moe import MoEExperts
from torchspec.utils.logging import logger


class V4EPMoEBlock(nn.Module):
    """V4 MoE block with Expert Parallel: V4 routing (sqrtsoftplus + correction bias)
    + TorchSpec EP-capable MoEExperts (grouped_mm + all-to-all when ep_group>1)
    + V4 shared expert. Drop-in for the decoder layer's `mlp` (forward(hidden, input_ids)).

    Reuses the validated TorchSpec EP path (moe_ep). NOTE: the routed-expert FFN uses
    plain SwiGLU (no swiglu_limit clamp) — a minor fidelity gap vs the native V4 experts;
    add clamp to MoEExperts if the MTP-init baseline acc regresses noticeably.
    """

    def __init__(self, config: DeepseekV4Config, ep_group=None):
        super().__init__()
        ep_group = ep_group if ep_group is not None else getattr(config, "ep_group", None)
        self.ep_group = ep_group
        self.hidden_size = config.hidden_size
        self.gate = DeepseekV4TopKRouter(config)  # V4 routing: sqrtsoftplus + correction bias
        self.experts = MoEExperts(
            config.num_local_experts, config.hidden_size, config.moe_intermediate_size,
            ep_group=ep_group,
        )
        self.shared_experts = DeepseekV4MLP(config)  # keeps swiglu_limit clamp

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, hid = hidden_states.shape
        flat = hidden_states.reshape(-1, hid)
        _logits, weights, indices = self.gate(flat)           # [N,top_k], [N,top_k]
        routed = self.experts(flat, weights.to(flat.dtype), indices).view(bsz, seq_len, hid)
        return routed + self.shared_experts(hidden_states)

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None


class Eagle3DeepseekV4ForCausalLM(Eagle3DraftModel):
    """DeepSeek-V4 native MTP/NextN draft, training-friendly for TorchSpec TTT.

    Structure (mirrors sglang DeepseekV4ModelNextN):
        embed_tokens -> enorm/hnorm -> e_proj/h_proj fuse -> [hc_mult streams]
        -> 1x DeepseekV4DecoderLayer (sliding MQA + mHC + MoE) -> hc_head collapse
        -> norm (shared_head.norm) -> lm_head
    """

    config_class = DeepseekV4Config

    def __init__(self, config: DeepseekV4Config, attention_backend: str = "sdpa", ep_group=None) -> None:
        super().__init__(config)
        self.ep_group = ep_group if ep_group is not None else getattr(config, "ep_group", None)
        # DeepSeek-V4 attention carries a per-head learnable sink (gpt-oss style) that
        # torch SDPA / flash do NOT support (see DeepseekV4PreTrainedModel: sdpa/flash
        # disabled). Use eager so the sink term is applied correctly.
        config._attn_implementation = "eager"

        # V4 MTP fuses a single target last-hidden with the token embedding (not 3 aux
        # hidden states like Eagle3); so the collected hidden is a single hidden_size vec.
        self.num_aux_hidden_states = 1

        self.target_vocab_size = config.vocab_size
        self.vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = config.rms_norm_eps

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # MTP front: separate norms + projections for token-embedding and prev-hidden
        self.enorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # NextN layer uses layer_idx 0 (sliding_attention, COMPRESS_RATIO_NEXTN_LAYER=0)
        self.midlayer = DeepseekV4DecoderLayer(config, layer_idx=0)
        # Replace the transformers SparseMoE with an EP-capable MoE block (TorchSpec
        # MoEExperts all-to-all when ep_group>1; V4 routing + shared expert preserved).
        self.midlayer.mlp = V4EPMoEBlock(config, ep_group=self.ep_group)

        # hc_head collapses the hc_mult residual streams; norm == shared_head.norm
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        # TorchSpec Eagle3 plumbing
        self.norm_output = False  # feed RAW (un-normed) hidden back; backbone applies hnorm
        self._cur_input_ids: Optional[torch.Tensor] = None

        if self.vocab_size != self.target_vocab_size:
            self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(self.vocab_size, dtype=torch.int64))

        # Modules built directly (not via from_pretrained) hold torch.empty params
        # (experts/router/sinks/hyper-connection) -> must initialize or forward NaNs.
        self.apply(self._init_v4_weights)

        logger.info(
            f"Eagle3DeepseekV4ForCausalLM: hidden={config.hidden_size} "
            f"heads={config.num_attention_heads} head_dim={config.head_dim} "
            f"experts={getattr(config, 'num_local_experts', getattr(config, 'n_routed_experts', '?'))} "
            f"top_k={config.num_experts_per_tok} hc_mult={config.hc_mult} "
            f"layer_type={config.layer_types[0]} mlp_type={config.mlp_layer_types[0]} "
            f"vocab={self.vocab_size}/{self.target_vocab_size}"
        )

    @torch.no_grad()
    def _init_v4_weights(self, module: nn.Module) -> None:
        """Initialize torch.empty params (mirrors DeepseekV4PreTrainedModel._init_weights)."""
        # skip meta-device instantiation (used for shape inspection / lazy materialize)
        p = next(module.parameters(recurse=False), None)
        if p is not None and p.is_meta:
            return
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, DeepseekV4RMSNorm):
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
        elif isinstance(module, DeepseekV4TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)

    # --- Eagle3DraftModel interface ---
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # stash for the hash-MoE router (needs token ids) inside backbone
        self._cur_input_ids = input_ids
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # V4 MTP fuses (token_emb, prev_hidden) per-step inside backbone; the incoming
        # hidden is the target's last hidden (already hidden_size). Pass through.
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"V4 MTP draft expects last_hidden of size {self.hidden_size}, "
                f"got {hidden_states.size(-1)}. Collect with store_last_hidden_states=True."
            )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def get_lm_head_params(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        return self.norm.weight, self.lm_head.weight, self.rms_norm_eps

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # --- MTP fusion: e_proj(enorm(emb)) + h_proj(hnorm(prev_hidden)) ---
        fused = self.e_proj(self.enorm(input_embeds)) + self.h_proj(self.hnorm(hidden_states))
        bsz, seq_len, _ = fused.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=fused.device).unsqueeze(0)
        position_ids = position_ids.reshape(-1, seq_len)  # (1,S) or (B,S); rotary broadcasts

        # V4 attention is eager-only and adds an additive [B,1,S,S] mask to attn weights.
        # Eagle3Model passes a 2D (B,S) mask for the "eager" backend, a 4D mask for "sdpa".
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_mask_4d = attention_mask
        else:
            mask2d = attention_mask if attention_mask is not None else torch.ones(
                bsz, seq_len, device=fused.device, dtype=torch.long
            )
            attn_mask_4d = self.prepare_decoder_attention_mask(
                attention_mask=mask2d, hidden_states=fused,
                batch_size=bsz, seq_length=seq_len, past_key_values_length=0,
            )

        # expand into hc_mult parallel residual streams: [B, S, hc_mult, H]
        hs = fused.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

        pos_emb = {
            "main": self.rotary_emb(fused, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(fused, position_ids=position_ids, layer_type="compress"),
        }

        past = DynamicCache(config=self.config) if (use_cache and DynamicCache is not None) else None
        out = self.midlayer(
            hs,
            position_embeddings=pos_emb,
            position_ids=position_ids,
            attention_mask=attn_mask_4d,
            input_ids=self._cur_input_ids,
            past_key_values=past,
        )
        collapsed = self.hc_head(out)  # [B, S, H]
        return collapsed, None, None
