"""African Wildlife 데이터셋(4개 부류)에서 3개 부류만 골라 YOLO 학습용 데이터셋을 만든다.

원본: Ultralytics African Wildlife dataset (buffalo, elephant, rhino, zebra / AGPL-3.0)
      https://docs.ultralytics.com/datasets/detect/african-wildlife/
사용 부류: elephant, rhino, zebra  ->  새 번호 0, 1, 2

- 제외한 부류(buffalo)가 하나라도 들어 있는 이미지는 통째로 뺀다.
  (라벨만 지우면 사진 속 들소가 '배경'으로 학습되기 때문)
- 원본의 train / val / test 분할은 그대로 유지한다. test는 학습·검증에 쓰지 않는다.
- --per-class N 을 주면 train에서 부류별로 최대 N장만 무작위로 고른다 (CPU 학습 시간 단축용).

사용 예:
    python scripts/prepare_dataset.py --src datasets_raw --dst datasets/wildlife3 --per-class 150
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

SRC_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
KEEP = ["elephant", "rhino", "zebra"]                      # 새 번호 순서
REMAP = {old: KEEP.index(n) for old, n in SRC_NAMES.items() if n in KEEP}


def read_labels(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), parts[1:]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets_raw")
    ap.add_argument("--dst", default="datasets/wildlife3")
    ap.add_argument("--per-class", type=int, default=0, help="train에서 부류별 최대 이미지 수 (0 = 전부)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(args.seed)

    for split in ["train", "val", "test"]:
        items = []                                   # (이미지 경로, 새 라벨 줄, 대표 부류)
        for img in sorted((src / "images" / split).iterdir()):
            lab = src / "labels" / split / (img.stem + ".txt")
            rows = read_labels(lab) if lab.exists() else []
            if not rows or any(c not in REMAP for c, _ in rows):
                continue                              # 라벨 없음 또는 제외 부류 포함 -> 사용 안 함
            new = [f"{REMAP[c]} {' '.join(box)}" for c, box in rows]
            main_cls = Counter(REMAP[c] for c, _ in rows).most_common(1)[0][0]
            items.append((img, new, main_cls))

        if split == "train" and args.per_class > 0:
            picked = []
            for k in range(len(KEEP)):
                group = [it for it in items if it[2] == k]
                rng.shuffle(group)
                picked += group[: args.per_class]
            items = sorted(picked, key=lambda it: it[0].name)

        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        img_count, inst_count = Counter(), Counter()
        for img, new, _ in items:
            shutil.copy2(img, dst / "images" / split / img.name)
            (dst / "labels" / split / (img.stem + ".txt")).write_text("\n".join(new) + "\n")
            classes = [int(l.split()[0]) for l in new]
            for c in set(classes):
                img_count[KEEP[c]] += 1
            for c in classes:
                inst_count[KEEP[c]] += 1
        print(f"{split:5s}: 이미지 {len(items):4d}장 | 부류별 이미지 {dict(img_count)} | 부류별 객체 {dict(inst_count)}")

    yaml = (
        f"# elephant / rhino / zebra 3개 부류 (원본: Ultralytics African Wildlife, AGPL-3.0)\n"
        f"path: {dst.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(KEEP))
    )
    (dst / "wildlife3.yaml").write_text(yaml)
    print("데이터 설정 파일:", dst / "wildlife3.yaml")


if __name__ == "__main__":
    main()
