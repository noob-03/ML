"""학습에 쓰지 않은 test 분할로 모델 성능을 잰다.

1) fine-tuning 한 모델: 전체·부류별 Precision, Recall, mAP50, mAP50-95 + 혼동행렬·PR 곡선 그림
2) (선택) --baseline 을 주면, fine-tuning 하기 전 COCO 사전학습 모델이 같은 test 이미지에서
   elephant / zebra 를 얼마나 찾는지도 잰다. rhino 는 COCO 80개 부류에 없어서 원래 모델로는 찾을 수 없다.
   입력 크기마다 세 가지를 잰다.
   - coco_pretrained_<크기>                  : test 178장 전부 (rhino 라벨은 뺀 사본)
   - coco_pretrained_<크기>_no_rhino_images  : 코뿔소가 찍힌 사진을 뺀 나머지
                                               (두 결과의 elephant 정밀도 차이 = 코뿔소 사진이 만든 오탐의 영향)
   - coco_on_rhino_images[<크기>]            : 코뿔소 사진에서 신뢰도 0.25 이상으로 낸 박스를 부류별로 센 것

사용 예:
    python scripts/evaluate.py --weights runs/ft_full/weights/best.pt --data datasets/wildlife3/wildlife3.yaml \
        --name full --baseline yolov8n.pt
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

COCO_ID = {"elephant": 20, "zebra": 22}          # COCO 80부류 번호 (rhino 없음)


def summarize(metrics, names):
    box = metrics.box
    out = {
        "all": {"precision": box.mp, "recall": box.mr, "mAP50": box.map50, "mAP50-95": box.map},
        "per_class": {},
        "speed_ms_per_image": metrics.speed,
    }
    for i, c in enumerate(box.ap_class_index):
        out["per_class"][names[int(c)]] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]), "mAP50-95": float(box.ap[i]),
        }
    return out


def make_coco_label_copy(data_root: Path, work: Path, coco_names):
    """test 라벨을 COCO 번호로 바꾼 복사본 (rhino 라벨은 COCO에 대응 부류가 없어 제외)."""
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    (work / "labels").mkdir(parents=True)
    keep = {0: COCO_ID["elephant"], 2: COCO_ID["zebra"]}   # 우리 번호 -> COCO 번호
    for img in (data_root / "images" / "test").iterdir():
        shutil.copy2(img, work / "images" / img.name)
        lab = data_root / "labels" / "test" / (img.stem + ".txt")
        lines = []
        for line in lab.read_text().splitlines():
            c, *box = line.split()
            if int(c) in keep:
                lines.append(" ".join([str(keep[int(c)]), *box]))
        (work / "labels" / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml = work / "coco_test.yaml"
    yaml.write_text(f"path: {work.resolve()}\ntrain: images\nval: images\ntest: images\n"
                    "names:\n" + "".join(f"  {i}: {n}\n" for i, n in coco_names.items()))
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/wildlife3/wildlife3.yaml")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--name", default="full")
    ap.add_argument("--out", default="results")
    ap.add_argument("--baseline", default=None, help="COCO 사전학습 가중치 (예: yolov8n.pt)")
    ap.add_argument("--baseline-imgsz", type=int, nargs="+", default=[320, 640],
                    help="사전학습 모델 평가 입력 크기 (320 = fine-tuning 모델과 같은 조건, 640 = 원래 학습 크기)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()          # 상대 경로면 Ultralytics가 runs/detect 아래로 옮기므로 절대 경로로
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    model = YOLO(args.weights)
    m = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=16, plots=True,
                  project=str(out_dir / "val_runs"), name=f"test_{args.name}", exist_ok=True)
    report["finetuned"] = summarize(m, model.names)

    if args.baseline:
        base = YOLO(args.baseline)
        data_root = Path(args.data).parent
        work = out_dir / "coco_label_test"
        coco_yaml = make_coco_label_copy(data_root, work, base.names)

        # 코뿔소(우리 번호 1)가 한 마리라도 찍힌 test 사진
        rhino_imgs = [p for p in sorted((data_root / "images" / "test").iterdir())
                      if any(l.split()[0] == "1" for l in
                             (data_root / "labels" / "test" / (p.stem + ".txt")).read_text().splitlines())]
        rhino_names = {p.name for p in rhino_imgs}
        no_rhino_list = work / "test_no_rhino_images.txt"
        no_rhino_list.write_text("".join(f"{p.resolve()}\n" for p in sorted((work / "images").iterdir())
                                         if p.name not in rhino_names))
        no_rhino_yaml = work / "coco_test_no_rhino.yaml"
        no_rhino_yaml.write_text(coco_yaml.read_text().replace("test: images", f"test: {no_rhino_list.name}"))

        # 코뿔소 사진을 뺀 같은 사진으로 fine-tuning 모델도 재서, COCO 모델과 elephant·zebra를 맞비교한다
        ft_list = work / "finetuned_test_no_rhino_images.txt"
        ft_list.write_text("".join(f"{p.resolve()}\n" for p in sorted((data_root / "images" / "test").iterdir())
                                   if p.name not in rhino_names))
        ft_yaml = work / "finetuned_test_no_rhino.yaml"
        ft_yaml.write_text(f"path: {data_root.resolve()}\ntrain: images/train\nval: images/val\ntest: {ft_list.resolve()}\n"
                           "names:\n" + "".join(f"  {i}: {n}\n" for i, n in model.names.items()))
        m = model.val(data=str(ft_yaml), split="test", imgsz=args.imgsz, batch=16, plots=False,
                      project=str(out_dir / "val_runs"), name=f"test_{args.name}_no_rhino_images", exist_ok=True)
        report["finetuned_no_rhino_images"] = summarize(m, model.names)
        report["finetuned_no_rhino_images"]["note"] = (f"imgsz={args.imgsz}. 코뿔소가 찍힌 사진 {len(rhino_imgs)}장을 뺀 "
                                                       f"test (coco_pretrained_*_no_rhino_images와 같은 사진)")

        n_test = len(list((data_root / "images" / "test").iterdir()))
        on_rhino = {}
        for sz in args.baseline_imgsz:
            for suffix, yaml, note in [
                ("", coco_yaml, f"imgsz={sz}. test {n_test}장. rhino는 COCO 부류에 없어 측정 대상에서 제외(탐지 불가)"),
                ("_no_rhino_images", no_rhino_yaml,
                 f"imgsz={sz}. 코뿔소가 찍힌 사진 {len(rhino_imgs)}장을 뺀 test {n_test - len(rhino_imgs)}장"),
            ]:
                key = f"coco_pretrained_{sz}{suffix}"
                b = base.val(data=str(yaml), split="test", imgsz=sz, batch=16, plots=False,
                             project=str(out_dir / "val_runs"), name=f"test_{key}", exist_ok=True)
                report[key] = summarize(b, base.names)
                report[key]["note"] = note

            # COCO 모델이 test의 코뿔소 사진을 무엇으로 착각하는지 (conf 0.25 이상 예측 부류 집계)
            boxes, images = Counter(), Counter()
            for p in rhino_imgs:
                r = base.predict(str(p), imgsz=sz, conf=0.25, verbose=False)[0]
                names = [base.names[int(c)] for c in r.boxes.cls]
                boxes.update(names)
                images.update(set(names))
            on_rhino[str(sz)] = {
                "rhino_images": len(rhino_imgs), "imgsz": sz, "conf": 0.25,
                "boxes_by_class": dict(boxes.most_common()),
                "images_containing_class": dict(images.most_common())}
        report["coco_on_rhino_images"] = on_rhino

    path = out_dir / f"test_metrics_{args.name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("저장:", path)


if __name__ == "__main__":
    main()
