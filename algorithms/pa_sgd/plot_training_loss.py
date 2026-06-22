"""
Vẽ đường cong loss của quá trình online training (post-by-post, snapshot-by-snapshot).

Đọc training_log.csv (sinh bởi train_pa_embeddings_dataset.py) và lưu ảnh PNG
loss curve vào cùng thư mục output.

Dùng standalone:
  python plot_training_loss.py --log outputs/x_embeddings_model/training_log.csv

Hoặc import:
  from plot_training_loss import plot_loss
  plot_loss("outputs/x_embeddings_model/training_log.csv")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend không cần GUI → chạy được trên server/headless
import matplotlib.pyplot as plt


def plot_loss(log_path: str, out_path: str | None = None, window: int = 200) -> str:
    """
    Vẽ MỘT total loss curve từ training_log.csv.

    Model đa nhiệm (4 reg head + 1 cls head) được tối ưu bằng 1 total loss
    (uncertainty weighting, Kendall 2018):
        L_total = (1/2σ₁²)·L_reg + (1/σ₂²)·L_cls + log σ₁ + log σ₂

    Panel trên : L_total (đường loss DUY NHẤT) — đường nhạt = raw, đậm = rolling-mean.
                 Vẽ kèm L_reg, L_cls mờ để thấy đóng góp từng thành phần.
    Panel dưới : σ_reg, σ_cls — trọng số bất định học được online.

    Parameters
    ----------
    log_path : str
        Đường dẫn training_log.csv (cột: step, l_total, l_reg, l_cls,
        sigma_reg, sigma_cls, ae_*).
    out_path : str | None
        Nơi lưu PNG. Mặc định: cùng thư mục, tên 'training_loss.png'.
    window : int
        Cửa sổ rolling-mean để làm mượt (loss online rất nhiễu theo từng sample).

    Returns
    -------
    str : đường dẫn file ảnh đã lưu.
    """
    log_path = Path(log_path)
    df = pd.read_csv(log_path)

    # Bỏ sample warm-start đầu (loss = NaN vì model chưa học gì để predict)
    df = df.dropna(subset=["l_total"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Không có dữ liệu loss hợp lệ trong {log_path}")

    step = df["step"].values
    w = max(1, min(window, len(df)))

    def smooth(col: str) -> np.ndarray:
        return df[col].rolling(window=w, min_periods=1).mean().values

    if out_path is None:
        out_path = log_path.parent / "training_loss.png"
    out_path = Path(out_path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # --- (1) TOTAL LOSS — đường loss duy nhất ---
    ax = axes[0]
    ax.plot(step, df["l_total"], color="tab:purple", alpha=0.15, linewidth=0.5)
    ax.plot(step, smooth("l_total"), color="tab:purple", linewidth=2.2,
            label=f"L_total (rolling-{w})")
    ax.plot(step, smooth("l_reg"), color="tab:blue", linewidth=1.2, alpha=0.7,
            label=f"└ L_reg (rolling-{w})")
    ax.plot(step, smooth("l_cls"), color="tab:green", linewidth=1.2, alpha=0.7,
            label=f"└ L_cls (rolling-{w})")
    # Giới hạn trục y theo đường đã làm mượt (spike warm-up rất lớn sẽ bị clip)
    ymax = max(np.nanmax(smooth("l_total")), np.nanmax(smooth("l_reg"))) * 1.3
    ax.set_ylim(0, max(ymax, 0.1))
    ax.set_ylabel("Total loss")
    ax.set_title("Online training — single multi-task total loss "
                 "(uncertainty weighting, Kendall 2018)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- (2) σ — trọng số bất định học được ---
    ax = axes[1]
    ax.plot(step, df["sigma_reg"], color="tab:blue", linewidth=1.6, label="σ_reg")
    ax.plot(step, df["sigma_cls"], color="tab:green", linewidth=1.6, label="σ_cls")
    ax.set_ylabel("Learned σ")
    ax.set_xlabel("Online step (snapshot transition)")
    ax.set_title("Uncertainty weights (σ) — task nhiễu hơn → σ lớn hơn → weight nhỏ hơn")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Vẽ loss curve từ training_log.csv")
    parser.add_argument("--log", default="outputs/x_embeddings_model/training_log.csv",
                        help="Đường dẫn training_log.csv")
    parser.add_argument("--out", default=None, help="Đường dẫn PNG output")
    parser.add_argument("--window", type=int, default=200,
                        help="Cửa sổ rolling-mean làm mượt")
    args = parser.parse_args()

    path = plot_loss(args.log, args.out, args.window)
    print(f"Saved loss curve: {path}")


if __name__ == "__main__":
    main()
