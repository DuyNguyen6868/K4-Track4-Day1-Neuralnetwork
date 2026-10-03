"""model.py — MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}

INITS = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:                      # q = 0: không chèn lớp Dropout nào
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))   # lớp ra: logit thô
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0, trừ "default").

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_, tức Var[W] = 2/(n_in+n_out)  (KHÔNG phải 1/n_in của slide;
                    hai công thức trùng nhau khi n_in = n_out)
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu"), Var[W] = 2/n_in
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear: W, b ~ U(-1/sqrt(n_in), 1/sqrt(n_in)),
                    tức Var[W] = 1/(3 n_in); KHÔNG phải He)
    """
    if init not in INITS:
        raise ValueError(f"init phải thuộc {INITS}, nhận được {init!r}")
    if init == "default":
        return
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(m.weight)
        elif init == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt theo lớp (ở bước 0, một lô val) — dùng cho thí nghiệm khởi tạo.

    Quy ước: với mỗi lớp ẩn ghi std SAU ReLU; với lớp ra ghi std của logit.
    Vậy M-base trả về 3 số: [std sau ReLU 1, std sau ReLU 2, std logit].
    """
    was_training = model.training
    model.eval()
    stats, h = [], x
    layers = list(model.net)
    for i, layer in enumerate(layers):
        h = layer(h)
        is_last = i == len(layers) - 1
        if isinstance(layer, nn.ReLU) or (is_last and isinstance(layer, nn.Linear)):
            stats.append(h.float().std().item())
    model.train(was_training)
    return stats
