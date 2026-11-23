"""실제 사용 기준(신뢰도 0.25, IoU 0.5)에서 test 분할의 혼동행렬을 계산한다.

Ultralytics(8.4.215) val의 혼동행렬은 mAP 계산용 설정 그대로 신뢰도 0.001 이상 예측을 모두 세기 때문에
'배경' 칸에 낮은 신뢰도 예측이 쌓인다(최종 모델 test 기준 2,407개). 영상 추론과 같은 조건(conf 0.25)으로 다시 센다.

매칭: 이미지마다 예측을 신뢰도 높은 순으로 보며, 아직 짝이 없는 정답 박스 중 IoU가 가장 큰 것(0.5 이상)과 짝짓는다.
  - 짝지어진 경우: (정답 부류, 예측 부류)
  - 짝이 없는 예측: (배경, 예측 부류)  -> 오탐
  - 짝이 없는 정답: (정답 부류, 배경)  -> 미탐

--misses-out 을 주면 놓친 정답 박스를 한 줄씩 적는다: 사진 이름, 부류, 박스 넓이(사진 대비 비율),
그 박스가 같은 사진의 다른 정답 박스에 덮인 최대 비율(겹쳐 선 동물인지 보려고).

사용 예:
    python scripts/confusion.py --weights weights/best.pt --data-root datasets/wildlife3 \
        --out results/confusion_conf025_freeze10.csv --misses-out results/confusion_misses_freeze10.csv
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
    ap.add_argument("--misses-out", default=None, help="놓친 정답 박스 목록 CSV (선택)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    labels = names + ["background"]
    bg = len(names)
    mat = np.zeros((len(labels), len(labels)), dtype=int)   # 행 = 정답, 열 = 예측

    misses = []                                                  # 놓친 정답 박스
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
            b = gt_box[j]
            area = (b[2] - b[0]) * (b[3] - b[1])
            others = np.delete(gt_box, j, axis=0)
            inter = (np.clip(np.minimum(b[2], others[:, 2]) - np.maximum(b[0], others[:, 0]), 0, None) *
                     np.clip(np.minimum(b[3], others[:, 3]) - np.maximum(b[1], others[:, 1]), 0, None))
            misses.append([img.name, names[gt_cls[j]], round(area / (w * h), 4),
                           round(float(inter.max() / area) if len(others) else 0.0, 2)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            wr.writerow([lab] + mat[i].tolist())
    if args.misses_out:
        with open(args.misses_out, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["image", "class", "box_area_ratio", "max_covered_by_other_gt"])
            wr.writerows(misses)
        crowded = sum(m[3] >= 0.5 for m in misses)
        print(f"놓친 정답 {len(misses)}개: 다른 정답 박스에 절반 이상 덮인 것 {crowded}개, "
              f"나머지 중 넓이 1% 미만 {sum(m[3] < 0.5 and m[2] < 0.01 for m in misses)}개 -> {args.misses_out}")
    print(f"conf={args.conf}, IoU={args.iou}, imgsz={args.imgsz}  (행: 정답, 열: 예측)")
    print("            " + " ".join(f"{l:>10s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:>10s}  " + " ".join(f"{v:>10d}" for v in mat[i]))


if __name__ == "__main__":
    main()
