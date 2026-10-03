"""optimizer.py — chọn bộ tối ưu (torch.optim), bộ lập lịch lr và cắt gradient.

Gom vào một chỗ để `train.py` gọn và mọi thí nghiệm dùng chung đúng một cách dựng optimizer.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của SGD/Adam là L2 trộn vào gradient (g <- g + wd*w, với Adam còn bị chia
    cho sqrt(v_hat)); weight_decay của AdamW là suy giảm tách riêng, không đi qua m, v.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận được {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch tốc độ học, gọi .step() sau MỖI bước cập nhật.

    name: None (lr hằng) | "cosine" (CosineAnnealingLR từ lr ban đầu về eta_min trong total_steps bước).
    Thí nghiệm nào dùng scheduler thì ghi vào cột notes của bảng.
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0))
    raise ValueError(f"scheduler phải là None hoặc 'cosine', nhận được {name!r}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm=None: chỉ đo chuẩn, không cắt (clip_grad_norm_ với max_norm=inf không đổi gradient).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này, nếu không
    chuẩn đo được đã bị nhân hệ số scale.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(params, math.inf if max_norm is None else max_norm)
    return float(total_norm)
