"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10        # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
TRAIN_SUBSET = 50_000  # số mẫu train CỐ ĐỊNH dùng để đo train loss ở chế độ eval()


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_train, y_train), (X_eval, y_eval)):
        assert X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == N_FEATURES, (X.dtype, X.shape)
        assert y.dtype == np.int64 and y.shape == (len(X),), (y.dtype, y.shape)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1, "nhãn phải nằm trong 0..6"
    assert eval_row_id.shape == (len(X_eval),)
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val. Dùng CÙNG seed và val_fraction cho mọi thí nghiệm.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Không tính trên val/eval: thống kê của các tập đó sẽ rò rỉ vào mô hình, làm điểm val/eval
    lạc quan hơn so với khi gặp dữ liệu thật sự mới.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)   # float64 để tránh sai số cộng dồn trên ~370k mẫu
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    safe_std = np.where(std > 0, std, 1.0).astype(np.float32)   # cột hằng: chỉ trừ mean, không chia 0
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / safe_std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (X float32, y int64)
        X_tr_sub, y_tr_sub   — tập con CỐ ĐỊNH của train (TRAIN_SUBSET mẫu) để đo train loss ở eval()
    và các giá trị numpy/python: eval_row_id, mean, std, majority_class, majority_val_acc
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)                      # CHỈ từ phần train còn lại
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    # tập con cố định của train: cùng một bộ chỉ số cho mọi thí nghiệm (không phụ thuộc seed thí nghiệm)
    sub_idx = np.random.default_rng(0).choice(len(X_tr), size=min(TRAIN_SUBSET, len(X_tr)), replace=False)

    majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())   # lớp đa số xác định trên TRAIN
    majority_val_acc = float((y_val == majority).mean())

    def t(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=t(X_tr, torch.float32), y_tr=t(y_tr, torch.int64),
        X_val=t(X_val, torch.float32), y_val=t(y_val, torch.int64),
        X_eval=t(X_eval, torch.float32), y_eval=t(y_eval, torch.int64),
        X_tr_sub=t(X_tr[sub_idx], torch.float32), y_tr_sub=t(y_tr[sub_idx], torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
        majority_class=majority, majority_val_acc=majority_val_acc,
    )
    if verbose:
        print(f"train (sau khi tách val): {tuple(data['X_tr'].shape)}   val: {tuple(data['X_val'].shape)}"
              f"   eval: {tuple(data['X_eval'].shape)}   device: {device}")
        print(f"luôn đoán lớp đa số (lớp {majority}) -> accuracy trên val = {majority_val_acc:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Lô cuối có thể nhỏ hơn batch_size và VẪN được dùng (không bỏ): mỗi epoch đi qua đủ mọi mẫu,
    số bước mỗi epoch = ceil(N / batch_size).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
