"""plots.py — ảnh biểu đồ là sản phẩm nộp (README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Bảng màu phân loại (thứ tự cố định, tối đa 8 đường trên một trục) và màu chữ/lưới
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e4e3df", "#fcfcfb"

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "grad_norm trung bình epoch (trước clip)",
    "grad_norm_max": "grad_norm lớn nhất trong epoch (trước clip)", "epoch_time_s": "thời gian / epoch (s)",
}


def _style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def _finite(values) -> np.ndarray:
    """None/NaN/inf -> NaN để matplotlib tự ngắt đường (lần chạy bị diverged)."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def cfg_label(cfg: dict) -> str:
    """Chuỗi cấu hình ngắn gọn để ghi vào tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    parts = [f"loss={cfg['loss']}", f"opt={cfg['optimizer']}", f"lr={cfg['lr']:g}", f"wd={cfg['weight_decay']:g}",
             f"batch={cfg['batch']}", f"epochs={cfg['epochs']}", f"hidden={hidden}", f"dropout={cfg['dropout']:g}",
             f"clip={'none' if cfg['clip_norm'] is None else format(cfg['clip_norm'], 'g')}",
             f"{cfg['precision']}", f"init={cfg['init']}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return "  ".join(parts)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch (cùng thang 0..1)
         (3) grad_norm theo epoch, đo TRƯỚC khi clip: trung bình và lớn nhất trong epoch
    Tiêu đề ghi exp_id và cấu hình; đường đứng nét đứt đánh dấu best_epoch.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    ax = axes[0]
    ax.plot(ep, _finite(h["train_loss"]), color=SERIES[0], lw=2, marker="o", ms=3, label="train loss (eval mode)")
    ax.plot(ep, _finite(h["val_loss"]), color=SERIES[1], lw=2, marker="o", ms=3, label="val loss")
    _style(ax, "epoch", f"loss ({cfg['loss'].upper()})", "Loss")

    ax = axes[1]
    ax.plot(ep, _finite(h["val_acc"]), color=SERIES[0], lw=2, marker="o", ms=3, label="val accuracy")
    ax.plot(ep, _finite(h["val_macro_f1"]), color=SERIES[1], lw=2, marker="o", ms=3, label="val macro-F1")
    _style(ax, "epoch", "điểm (0–1)", "Val accuracy / macro-F1")

    ax = axes[2]
    gmean, gmax = _finite(h["grad_norm"]), _finite(h["grad_norm_max"])
    ax.plot(ep, gmean, color=SERIES[0], lw=2, marker="o", ms=3, label="trung bình epoch")
    ax.plot(ep, gmax, color=SERIES[1], lw=1.5, ls=":", marker="o", ms=3, label="lớn nhất trong epoch")
    if cfg["clip_norm"] is not None:
        ax.axhline(cfg["clip_norm"], color=MUTED, lw=1.2, ls="-.", label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    _style(ax, "epoch", "‖g‖₂ toàn cục", "grad_norm (trước clip)")
    pos = gmax[np.isfinite(gmax) & (gmax > 0)]
    if len(pos) and pos.max() / max(np.nanmin(gmean[gmean > 0]) if np.any(gmean > 0) else pos.min(), 1e-12) > 50:
        ax.set_yscale("log")      # gai gradient lớn: thang log để còn thấy đường trung bình

    for ax in axes:
        if s["best_epoch"] is not None:
            ax.axvline(s["best_epoch"], color=MUTED, lw=1, ls="--", label=f"best epoch = {s['best_epoch']}")
        ax.legend(fontsize=8, frameon=False)

    if s["diverged"]:
        status = "DIVERGED (loss NaN/inf)"
    else:
        status = f"best val_loss={s['best_val_loss']:.4f}  val_acc={s['val_acc']:.4f}  val_macro_f1={s['val_macro_f1']:.4f}"
    fig.suptitle(f"{cfg['exp_id']}  [{cfg['group']}]  —  {status}\n{cfg_label(cfg)}",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, path)


def plot_compare(results: list[dict], metric, path: str, title: str = "", labels: list[str] | None = None) -> None:
    """Vẽ chồng một hoặc nhiều chỉ số của nhiều thí nghiệm: mỗi chỉ số một ô, mỗi thí nghiệm một đường,
    chú thích bằng exp_id. `metric` là chuỗi hoặc danh sách chuỗi (ví dụ ["val_loss", "val_macro_f1", "grad_norm"]).
    Tối đa 8 thí nghiệm trên một ảnh (bảng màu cố định, không quay vòng).
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    if len(results) > len(SERIES):
        raise ValueError(f"tối đa {len(SERIES)} đường trên một ảnh; hãy tách nhóm")
    labels = labels or [r["cfg"]["exp_id"] for r in results]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.3), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        all_vals = []
        for r, lab, color in zip(results, labels, SERIES):
            vals = _finite(r["history"][m])
            all_vals.append(vals)
            ax.plot(r["history"]["epoch"], vals, color=color, lw=2, marker="o", ms=3, label=lab)
        _style(ax, "epoch", METRIC_LABELS.get(m, m), METRIC_LABELS.get(m, m))
        flat = np.concatenate(all_vals) if all_vals else np.array([])
        flat = flat[np.isfinite(flat) & (flat > 0)]
        if m.startswith("grad_norm") and len(flat) and flat.max() / flat.min() > 50:
            ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    _save(fig, path)


def plot_lr_sweep(groups: dict[str, list[dict]], metric: str, path: str, title: str = "") -> None:
    """Ảnh nhiều ô nhỏ: mỗi bộ tối ưu một ô, mỗi lr một đường, chung trục y để so trực tiếp.

    groups: {"tên bộ tối ưu": [result, ...]}.
    """
    names = list(groups)
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], names):
        for r, color in zip(groups[name], SERIES):
            ax.plot(r["history"]["epoch"], _finite(r["history"][metric]), color=color, lw=2, marker="o", ms=3,
                    label=f"lr={r['cfg']['lr']:g}" + (" (diverged)" if r["summary"]["diverged"] else ""))
        _style(ax, "epoch", METRIC_LABELS.get(metric, metric) if ax is axes[0][0] else "", name)
        ax.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    _save(fig, path)


def plot_curve(values, path: str, xlabel: str, ylabel: str, title: str, logy: bool = False) -> None:
    """Một đường đơn (ví dụ loss theo bước khi quá khớp 20 mẫu)."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(range(1, len(values) + 1), _finite(values), color=SERIES[0], lw=2)
    _style(ax, xlabel, ylabel, title)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, path)


def plot_confusion(cm, path: str, title: str = "") -> None:
    """Ma trận nhầm lẫn chuẩn hoá theo hàng (recall); mỗi ô ghi tỉ lệ và số mẫu."""
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    k = cm.shape[0]
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{frac[i, j]:.2f}\n{int(cm[i, j])}", ha="center", va="center", fontsize=7.5,
                    color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlabel("lớp dự đoán", color=INK)
    ax.set_ylabel("lớp thật", color=INK)
    ax.set_title(title or "Ma trận nhầm lẫn (chuẩn hoá theo hàng)", color=INK, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="tỉ lệ trong lớp thật (recall trên đường chéo)")
    fig.tight_layout()
    _save(fig, path)
