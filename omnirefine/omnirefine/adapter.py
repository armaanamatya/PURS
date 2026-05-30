"""Qwen2.5-Omni integration boundary (the /system-design seam).

This is the ONLY module that should know about the model.  OmniRefine is
training-free, so every similarity / saliency signal it consumes comes from
a *partial forward pass* at a probe layer L.  The control flow (mirrors how
OmniZip-style methods hook the model) is:

    1. Run prefill up to layer L (cfg.layer_probe).
    2. Read hidden states H_L for all multimodal tokens and the layer-L
       attention.  Build the native chunk indexing from Qwen2.5-Omni's
       temporal position indices (fixed-duration interleaved audio/video
       chunks, Sec 3.1).
    3. Assemble a `ProbeInputs` and call `omnirefine.pipeline.compress`.
    4. Translate the returned `KeepMask` into a token keep-set, prune the KV
       cache / token sequence (and write back merged representations), then
       continue prefill from layer L+1 with the compressed sequence.

Nothing below performs step 1/2/4 -- those are model-version specific and
are intentionally left as documented stubs (raise NotImplementedError).  The
testable algorithm lives entirely in `pipeline.compress`.

See DESIGN.md for the full data-flow diagram and REPRODUCTION_NOTES.md for
the choices the paper leaves open (the probe layer L is the load-bearing
one).
"""
from __future__ import annotations

from typing import Any

from .config import OmniRefineConfig
from .pipeline import KeepMask, ProbeInputs, compress


class OmniRefineAdapter:
    """Abstract adapter; subclass per model checkpoint / runtime."""

    def __init__(self, cfg: OmniRefineConfig | None = None) -> None:
        self.cfg = cfg or OmniRefineConfig()

    # ---- Step 1+2: produce ProbeInputs from a partial forward pass ----
    def probe(self, model: Any, model_inputs: Any) -> ProbeInputs:
        """Run prefill to layer L and extract hidden states + saliency.

        Must populate, at probe layer ``self.cfg.layer_probe``:
          * video hidden grids per frame + global token ids + native chunk ids
          * audio hidden states + global token ids + native chunk ids
          * audio saliency = fused attention-based importance (PARTIAL: the
            paper does not fix which attention; a reasonable choice is the
            mean attention mass each audio token receives at layer L).
        """
        raise NotImplementedError(
            "Implement probe() for your Qwen2.5-Omni runtime: run prefill to "
            "cfg.layer_probe, read H_L + attention, build native chunk indices."
        )

    # ---- Step 3: the model-free core ----
    def plan(self, inputs: ProbeInputs) -> KeepMask:
        return compress(inputs, self.cfg)

    # ---- Step 4: apply the plan to the running model ----
    def apply(self, model: Any, keep: KeepMask) -> Any:
        """Prune tokens / KV cache to the keep-set, write back merged reps,
        and continue prefill from layer L+1."""
        raise NotImplementedError(
            "Implement apply() for your runtime: gather kept token indices, "
            "prune the KV cache, scatter merged representations, resume prefill."
        )

    # ---- Convenience: full run (requires probe/apply to be implemented) ----
    def run(self, model: Any, model_inputs: Any) -> Any:
        inputs = self.probe(model, model_inputs)
        keep = self.plan(inputs)
        return self.apply(model, keep)
