"""
استخراج وزن‌های توجه (Attention) از مدل‌های GNN/Transformer

ابزاری model-agnostic که بدون نیاز به تغییر معماری مدل، وزن‌های لایه‌های Attention را
از طریق forward hook در حین یک پیش-رو (forward pass) واقعی می‌گیرد. برای استفاده در
واحدهای ۲ و ۳ (MPNN+Attention، Transformer+GNN) که هنوز پیاده‌سازی نشده‌اند طراحی شده،
اما با هر ``torch.nn.Module`` که یک زیرماژول attention-مانند دارد کار می‌کند.

See docs/SRS.md §4.7 (FR-09).
"""

from typing import Dict, List

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "پکیج 'torch' نصب نیست. با «pip install torch» یا از طریق requirements.txt نصب کنید."
    ) from exc

import numpy as np


class AttentionExtractor:
    """
    ثبت forward hook روی یک یا چند لایه از مدل و گرفتن خروجی وزن‌های Attention آن‌ها
    در حین اجرای واقعی مدل، بدون تغییر کد مدل.

    مثال:
        with AttentionExtractor(model, ["gnn.attn_layer"]) as extractor:
            attn_weights = extractor.extract(batch)
    """

    def __init__(self, model: "nn.Module", attention_layer_names: List[str]):
        if not attention_layer_names:
            raise ValueError("attention_layer_names نباید خالی باشد")

        self.model = model
        self._captured: Dict[str, "torch.Tensor"] = {}
        self._handles = []

        named_modules = dict(model.named_modules())
        for name in attention_layer_names:
            module = named_modules.get(name)
            if module is None:
                raise ValueError(
                    f"لایه '{name}' در مدل یافت نشد. لایه‌های موجود: {list(named_modules)}"
                )
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # بسیاری از لایه‌های Attention یک تاپل (خروجی، وزن‌های attention) برمی‌گردانند
            attn = output[1] if isinstance(output, tuple) and len(output) > 1 else output
            self._captured[name] = attn.detach()

        return hook

    def extract(self, *forward_args, **forward_kwargs) -> Dict[str, np.ndarray]:
        """اجرای یک forward pass واقعی و بازگرداندن وزن‌های attention گرفته‌شده."""
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
                "هیچ وزن attention‌ای گرفته نشد — بررسی کنید نام لایه‌ها درست باشد و "
                "خروجی forward آن‌ها شامل وزن‌های attention باشد."
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
