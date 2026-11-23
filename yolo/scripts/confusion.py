"""실제 사용 기준(신뢰도 0.25, IoU 0.5)에서 test 분할의 혼동행렬을 계산한다.

Ultralytics val의 혼동행렬은 mAP 계산용으로 신뢰도 0.001 이상 예측을 모두 세기 때문에
'배경' 칸에 낮은 신뢰도 예측이 수천 개 들어간다. 영상 추론과 같은 조건(conf 0.25)으로 다시 센다.

매칭: 이미지마다 예측을 신뢰도 높은 순으로 보며, 아직 짝이 없는 정답 박스 중 IoU가 가장 큰 것(0.5 이상)과 짝짓는다.
  - 짝지어진 경우: (정답 부류, 예측 부류)
  - 짝이 없는 예측: (배경, 예측 부류)  -> 오탐
  - 짝이 없는 정답: (정답 부류, 배경)  -> 미탐

사용 예:
    python scripts/confusion.py --weights weights/best.pt --data-root datasets/wildlife3 --out results/confusion_conf025.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data-root", default="datasets/wildlife3")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="results/confusion_conf025.csv")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    labels = names + ["background"]
    bg = len(names)
    mat = np.zeros((len(labels), len(labels)), dtype=int)   # 행 = 정답, 열 = 예측

    root = Path(args.data_root)
    for img in sorted((root / "images" / "test").iterdir()):
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        h, w = res.orig_shape
        gts = []
        for line in (root / "labels" / "test" / (img.stem + ".txt")).read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            gts.append([int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        gt_cls = np.array([g[0] for g in gts], dtype=int)
        gt_box = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
        used = np.zeros(len(gts), dtype=bool)
        order = np.argsort(-res.boxes.conf.numpy())
        for k in order:
            pb, pc = res.boxes.xyxy.numpy()[k], int(res.boxes.cls.numpy()[k])
            if len(gts):
                ious = iou(pb, gt_box)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= args.iou:
                    used[j] = True
                    mat[gt_cls[j], pc] += 1
                    continue
            mat[bg, pc] += 1                                    # 오탐
        for j in np.where(~used)[0]:
            mat[gt_cls[j], bg] += 1                             # 미탐

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            wr.writerow([lab] + mat[i].tolist())
    print(f"conf={args.conf}, IoU={args.iou}, imgsz={args.imgsz}  (행: 정답, 열: 예측)")
    print("            " + " ".join(f"{l:>10s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:>10s}  " + " ".join(f"{v:>10d}" for v in mat[i]))


if __name__ == "__main__":
    main()
