"""
Extraction of attention weights from GNN/Transformer models

A model-agnostic tool that captures the weights of Attention layers via a forward hook
during a real forward pass, without requiring changes to the model architecture. Designed for use in
Units 2 and 3 (MPNN+Attention, Transformer+GNN) that are not yet implemented,
but it works with any ``torch.nn.Module`` that has an attention-like submodule.

See docs/SRS.md §4.7 (FR-09).
"""

from typing import Dict, List

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Package 'torch' is not installed. Install it with «pip install torch» or via requirements.txt."
    ) from exc

import numpy as np


class AttentionExtractor:
    """
    Register a forward hook on one or more layers of the model and capture their Attention weights
    during a real model run, without changing the model code.

    Example:
        with AttentionExtractor(model, ["gnn.attn_layer"]) as extractor:
            attn_weights = extractor.extract(batch)
    """

    def __init__(self, model: "nn.Module", attention_layer_names: List[str]):
        if not attention_layer_names:
            raise ValueError("attention_layer_names must not be empty")

        self.model = model
        self._captured: Dict[str, "torch.Tensor"] = {}
        self._handles = []

        named_modules = dict(model.named_modules())
        for name in attention_layer_names:
            module = named_modules.get(name)
            if module is None:
                raise ValueError(
                    f"Layer '{name}' not found in the model. Available layers: {list(named_modules)}"
                )
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # Many Attention layers return a tuple (output, attention weights)
            attn = output[1] if isinstance(output, tuple) and len(output) > 1 else output
            self._captured[name] = attn.detach()

        return hook

    def extract(self, *forward_args, **forward_kwargs) -> Dict[str, np.ndarray]:
        """Run a real forward pass and return the captured attention weights."""
        self._captured.clear()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(*forward_args, **forward_kwargs)
        finally:
            self.model.train(was_training)

        if not self._captured:
            raise RuntimeError(
                "No attention weights were captured — check that the layer names are correct and "
                "that their forward output includes attention weights."
            )
        return {name: tensor.cpu().numpy() for name, tensor in self._captured.items()}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "AttentionExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
