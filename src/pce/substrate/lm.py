"""Local-LM substrate: thin wrapper around the chosen HuggingFace causal LM.

The substrate is the `cit`-layer's physical anchor. It exposes:

* `generate(...)`: produces a single Candidate under (tau, top_p, top_k, seed)
* `score(...)`:    re-scores a token sequence under the LM, returning `sum log p`
* `entropy_at(...)`: per-step entropy of the LM's next-token distribution

The implementation uses `transformers.AutoModelForCausalLM` directly (we control
the sampling loop ourselves so the temperature monotonicity invariant in
`tests/operators/test_cit.py` is exact).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from pce.substrate.embed import Embedder
from pce.types import Candidate

DEFAULT_LM_ID = "Qwen/Qwen2-1.5B-Instruct"


def _autodetect_device() -> str:
    """Pick the fastest available torch backend.

    Order: CUDA -> MPS (Apple Silicon) -> CPU. Honour PCE_DEVICE if set so tests
    can pin a specific backend.
    """
    import os

    forced = os.environ.get("PCE_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autodetect_dtype(device: str) -> str:
    import os

    forced = os.environ.get("PCE_DTYPE", "").strip().lower()
    if forced:
        return str(forced)
    # On MPS, fp16 is meaningfully faster and numerically fine for inference.
    if device == "mps":
        return "float16"
    if device == "cuda":
        return "float16"
    return "float32"


@dataclass(frozen=True)
class LMConfig:
    model_id: str = DEFAULT_LM_ID
    dtype: str = ""
    device: str = ""

    def resolved_device(self) -> str:
        return self.device or _autodetect_device()

    def resolved_dtype(self) -> str:
        dev = self.resolved_device()
        return self.dtype or _autodetect_dtype(dev)


@lru_cache(maxsize=2)
def _load_lm(model_id: str, dtype: str, device: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return tok, model


class LocalLM:
    """The cit-substrate. Holds the tokenizer + model + the embedder used to embed candidates.

    Implements `pce.substrate.lm_protocol.GeneratorProtocol` (alias `LMProtocol`);
    see ADR-004 for the pluggable-substrate contract. v0.3 (Phase 2) extended
    the protocol with three capability flags -- LocalLM exposes real logprobs
    so it advertises ``supports_logprobs=True``.
    """

    name: str = "qwen2-1.5b"
    supports_logprobs: bool = True
    supports_score: bool = False
    supports_entropy: bool = True

    def __init__(self, config: LMConfig | None = None, embedder: Embedder | None = None) -> None:
        cfg = config or LMConfig()
        device = cfg.resolved_device()
        dtype = cfg.resolved_dtype()
        self.config = LMConfig(model_id=cfg.model_id, dtype=dtype, device=device)
        self.tok, self.model = _load_lm(self.config.model_id, dtype, device)
        self._embedder = embedder or Embedder()
        self.eos_id: int = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else -1
        self.vocab_size: int = int(self.model.config.vocab_size)

    @torch.no_grad()
    def _next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids)
        logits: torch.Tensor = out.logits[:, -1, :].float()
        return logits

    def _apply_sampler(
        self,
        logits: torch.Tensor,
        *,
        tau: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Return a probability distribution after temperature + top-p + top-k filtering."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        scaled = logits / float(tau)
        if top_k > 0 and top_k < scaled.shape[-1]:
            kth, _ = torch.topk(scaled, k=top_k, dim=-1)
            threshold = kth[..., -1:]
            scaled = torch.where(scaled < threshold, torch.full_like(scaled, float("-inf")), scaled)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
        return torch.softmax(scaled, dim=-1)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        sampler: dict[str, float] | None = None,
        seed: int = 0,
    ) -> Candidate:
        sampler = dict(sampler or {})
        tau = float(sampler.get("tau", 1.0))
        top_p = float(sampler.get("top_p", 0.95))
        top_k = int(sampler.get("top_k", 50))

        # MPS does not support per-device torch.Generator; run sampling on CPU
        # for determinism while keeping the model graph on the configured device.
        gen_rng = torch.Generator(device="cpu").manual_seed(int(seed))

        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.config.device)
        generated: list[int] = []
        logp_sum = 0.0

        for _ in range(int(max_tokens)):
            logits = self._next_logits(input_ids)
            probs = self._apply_sampler(logits, tau=tau, top_p=top_p, top_k=top_k)
            probs_cpu = probs.detach().to("cpu").float()
            sampled_cpu = torch.multinomial(probs_cpu, num_samples=1, generator=gen_rng)
            sampled = sampled_cpu.to(self.config.device)
            next_id = int(sampled.item())
            chosen_p = float(probs[0, next_id].item())
            if chosen_p > 0:
                logp_sum += math.log(chosen_p)
            generated.append(next_id)
            input_ids = torch.cat([input_ids, sampled.view(1, 1)], dim=-1)
            if next_id == self.eos_id:
                break

        text_raw = self.tok.decode(generated, skip_special_tokens=True)
        text: str = text_raw if isinstance(text_raw, str) else str(text_raw)
        # Avoid degenerate empty completions when temperature is very low and EOS hits early.
        if not text.strip():
            text_raw2 = self.tok.decode(generated, skip_special_tokens=False)
            text = text_raw2 if isinstance(text_raw2, str) else str(text_raw2)
        embedding = self._embedder.encode(text or " ")

        return Candidate(
            seed=int(seed),
            sampler={"tau": tau, "top_p": top_p, "top_k": float(top_k)},
            tokens=tuple(generated),
            text=text,
            logp=float(logp_sum),
            embedding=embedding,
        )

    def entropy_at(self, prompt: str, *, tau: float = 1.0) -> float:
        """Shannon entropy (bits) of the next-token distribution at `prompt`'s end."""
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.config.device)
        logits = self._next_logits(input_ids)
        probs = torch.softmax(logits / float(tau), dim=-1)
        p = probs.clamp_min(1e-30)
        ent = -float((p * torch.log2(p)).sum().item())
        return ent

    def argmax_next(self, prompt: str) -> int:
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.config.device)
        logits = self._next_logits(input_ids)
        return int(torch.argmax(logits, dim=-1).item())

    def length_proxy_logp(self, candidate: Candidate) -> float:
        """Real log-probability is in ``candidate.logp`` for LocalLM (supports_logprobs=True).

        Provided so LocalLM satisfies :class:`pce.substrate.lm_protocol.GeneratorProtocol`
        even though the proxy is not the path callers normally take. Returns
        ``candidate.logp`` directly.
        """
        return float(candidate.logp)

    def report(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "supports_logprobs": self.supports_logprobs,
            "supports_score": self.supports_score,
            "supports_entropy": self.supports_entropy,
        }
