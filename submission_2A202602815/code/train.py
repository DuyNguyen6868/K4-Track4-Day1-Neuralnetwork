"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches, N_CLASSES
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` chọn bằng val trong notebook rồi truyền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine"
    seed=1,
)

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán. Cùng công thức với scripts/evaluate.py.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval() và fp32."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, đúng như nn.MSELoss: KHÔNG có hệ số 1/2 và lấy
               trung bình trên MỌI phần tử (B x 7), tức mỗi mẫu = (1/7) * sum_c (logit_c - onehot_c)^2.
       reduction="sum" trả về tổng loss theo MẪU (để cộng dồn qua các lô rồi chia N).
    """
    logits = logits.float()            # dưới autocast logit có thể là fp16/bf16; loss luôn tính ở fp32
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        per_sample = ((logits - F.one_hot(y, logits.shape[1]).float()) ** 2).mean(dim=1)
        return per_sample.mean() if reduction == "mean" else per_sample.sum()
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận được {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, confusion) ở chế độ eval() (dropout tắt) và no_grad.

    Dùng cho: train loss (trên tập con CỐ ĐỊNH của train), val. KHÔNG dùng cho eval để chọn cấu hình.
    """
    model.eval()
    total_loss, n = 0.0, len(X)
    cm = torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total_loss += compute_loss(logits, yb, loss_name, reduction="sum").item()
        pred = logits.argmax(dim=1)
        cm += torch.bincount(yb * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()       # hàng = thật, cột = dự đoán
    return dict(loss=total_loss / n, acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm), confusion=cm)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (khoá thiếu lấy từ DEFAULT_CFG)
        data: kết quả của data.prepare_data

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                     "grad_norm" (trung bình epoch, TRƯỚC clip), "grad_norm_max", "clip_frac", "epoch_time_s"},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1" (ở best_epoch), "time_per_epoch_s", "peak_mem_MB", "diverged",
                     + phụ: n_params, n_steps, grad_norm_p50/p90/max (theo BƯỚC), clip_frac, device, total_time_s},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}

    Hàm này KHÔNG bao giờ đọc X_eval / y_eval: chọn epoch và cấu hình chỉ bằng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    hidden = tuple(cfg["hidden"])
    cfg["hidden"] = hidden
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    X_sub, y_sub = data["X_tr_sub"], data["y_tr_sub"]
    device = X_tr.device
    loss_name, precision = cfg["loss"], cfg["precision"]

    # ---- 0. seed, model, optimizer, scheduler, AMP
    set_seed(cfg["seed"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS[hidden], f"số tham số {n_params} != {EXPECTED_PARAMS[hidden]}"
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], steps_per_epoch * cfg["epochs"])
    use_amp = precision != "fp32"
    if use_amp and precision not in AMP_DTYPES:
        raise ValueError(f"precision phải là fp32/fp16/bf16, nhận được {precision!r}")
    # FP16 cần GradScaler (nhân loss với s để gradient nhỏ không bị làm tròn về 0); BF16 thì không
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])                     # thứ tự xáo lô phụ thuộc seed thí nghiệm
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. loss bước 0 trên val, TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7 với CE)
    step0_loss = evaluate(model, X_val, y_val, loss_name)["loss"]

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    step_norms: list[float] = []          # grad_norm của TỪNG bước (trước clip), để chọn ngưỡng c
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged = False
    t_start = time.perf_counter()

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_norms: list[float] = []
        _sync(device)
        t0 = time.perf_counter()
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
            # autocast chỉ bọc forward + loss; tham số vẫn là fp32
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPES.get(precision), enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)        # luôn unscale: grad_norm ghi lại phải là chuẩn THẬT
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(optimizer)            # tự bỏ qua bước nếu gradient có inf/NaN (tràn FP16)
                scaler.update()
            else:
                if not math.isfinite(gn):
                    diverged = True
                    break
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if math.isfinite(gn):
                epoch_norms.append(gn)
        _sync(device)
        epoch_time = time.perf_counter() - t0     # chỉ tính vòng huấn luyện, không tính phần đo train/val

        # ---- cuối epoch: mọi thứ đo ở eval() để train loss và val loss cùng thang đo
        tr = evaluate(model, X_sub, y_sub, loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            diverged = True
        step_norms += epoch_norms
        c = cfg["clip_norm"]
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(epoch_norms)) if epoch_norms else float("nan"))
        hist["grad_norm_max"].append(float(np.max(epoch_norms)) if epoch_norms else float("nan"))
        hist["clip_frac"].append(float(np.mean(np.array(epoch_norms) > c)) if (c is not None and epoch_norms) else 0.0)
        hist["epoch_time_s"].append(epoch_time)
        if verbose:
            print(f"  [{cfg['exp_id']}] epoch {epoch:2d}/{cfg['epochs']}  train {tr['loss']:.4f}  val {va['loss']:.4f}"
                  f"  acc {va['acc']:.4f}  f1 {va['macro_f1']:.4f}  gnorm {hist['grad_norm'][-1]:.3f}  {epoch_time:.1f}s")
        if diverged:
            break
        if va["loss"] < best_val_loss:            # "dừng sớm": nhớ trọng số của epoch có val loss thấp nhất
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- 3. tóm tắt tại best_epoch
    bi = None if best_epoch is None else best_epoch - 1
    norms = np.array(step_norms) if step_norms else np.array([np.nan])
    c = cfg["clip_norm"]
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=None if bi is None else best_val_loss,
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1] if hist["train_loss"] else None,
        final_val_loss=hist["val_loss"][-1] if hist["val_loss"] else None,
        val_acc=None if bi is None else hist["val_acc"][bi],
        val_macro_f1=None if bi is None else hist["val_macro_f1"][bi],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])) if hist["epoch_time_s"] else None,
        peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        diverged=diverged,
        n_params=n_params, n_steps=len(step_norms), device=str(device),
        grad_norm_p50=float(np.nanpercentile(norms, 50)), grad_norm_p90=float(np.nanpercentile(norms, 90)),
        grad_norm_max=float(np.nanmax(norms)),
        clip_frac=float(np.mean(norms > c)) if c is not None else 0.0,
        total_time_s=time.perf_counter() - t_start,
    )
    if verbose:
        tail = "DIVERGED" if diverged else f"best epoch {best_epoch}: val_loss {best_val_loss:.4f}  " \
               f"acc {summary['val_acc']:.4f}  macro-F1 {summary['val_macro_f1']:.4f}"
        print(f"=> {cfg['exp_id']}: step0 {step0_loss:.4f} | {tail} | {summary['time_per_epoch_s']:.1f}s/epoch")
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best_state)


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred` (pred là số nguyên 0..6)."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert row_id.shape == preds.shape, "row_id và preds phải cùng độ dài"
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Sau đó chấm bằng run_official_eval (gọi scripts/evaluate.py). Trả về mảng dự đoán.
    """
    if result.get("best_state") is None:
        raise ValueError("result không có best_state (lần chạy bị diverged hoặc nạp từ cache)")
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"]).cpu().numpy()     # eval mode (không dropout), fp32
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds


def run_official_eval(pred_path: str, out_json: str, repo_root: str) -> dict:
    """Chạy scripts/evaluate.py (script chấm chính thức) và trả về nội dung eval_result.json."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}      # script in tiếng Việt; tránh lỗi cp1252 trên Windows
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
         "--out", os.path.abspath(out_json)],
        cwd=repo_root, env=env, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py lỗi:\n{proc.stdout}\n{proc.stderr}")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)
