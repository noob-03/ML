"""epoch별 검증 mAP50-95 학습 곡선 그림(results/learning_curves.png)을 학습 로그 CSV에서 다시 그린다.

- results/train_log_<방식>.csv          : 최종 비교에 쓴 시드 0 학습 (굵은 선)
- results/train_log_<방식>_s<시드>.csv  : 시드를 바꾼 반복 학습 (가는 선, 있으면 함께 그림)

사용 예:
    python scripts/plot_curves.py --logs results --out results/learning_curves.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"freeze10": ("#1f5fbf", "freeze10 (backbone frozen, final model)"),
         "full": ("#8c8c8c", "full (all layers trained)")}


def read_log(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return [int(r["epoch"]) for r in rows], [float(r["metrics/mAP50-95(B)"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results")
    ap.add_argument("--out", default="results/learning_curves.png")
    ap.add_argument("--close-mosaic", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    logs = Path(args.logs)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    n_epochs, seeds = 0, {}
    for mode, (color, label) in STYLE.items():
        extra = sorted(logs.glob(f"train_log_{mode}_s*.csv"))
        seeds[mode] = 1 + len(extra)
        for p in extra:
            e, v = read_log(p)
            ax.plot(e, v, color=color, lw=1, alpha=0.45)
        e, v = read_log(logs / f"train_log_{mode}.csv")
        n_epochs = max(n_epochs, max(e))
        ax.plot(e, v, color=color, lw=2.2, marker="o", ms=3.5, label=label)

    ax.axvspan(0.5, args.warmup + 0.5, color="#000000", alpha=0.04, lw=0)
    ax.axvspan(n_epochs - args.close_mosaic + 0.5, n_epochs + 0.5, color="#000000", alpha=0.07, lw=0)
    ax.text((args.warmup + 1) / 2, 0.81, "warm-up", ha="center", va="top", fontsize=9, color="#666666")
    ax.text(n_epochs - (args.close_mosaic - 1) / 2, 0.81, "mosaic off", ha="center", va="top", fontsize=9,
            color="#666666")
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.set_ylim(0.4, 0.82)
    ax.set_xticks(range(1, n_epochs + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP50-95")
    note = "" if max(seeds.values()) == 1 else \
        f"  -  thick: seed 0, thin: seeds 1-{max(seeds.values()) - 1}"
    ax.set_title(f"Validation mAP50-95 per epoch (YOLOv8n, 3 classes, imgsz 320){note}", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
