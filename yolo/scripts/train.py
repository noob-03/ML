"""COCO 사전학습 YOLOv8n을 elephant / rhino / zebra 3개 부류로 fine-tuning 한다.

두 가지 방식을 같은 조건으로 학습해 비교한다.
  full     : 모든 층을 다시 학습 (일반적인 fine-tuning)
  freeze10 : 백본(0~9번 층)을 고정하고 넥·헤드만 학습
             (CV-4 슬라이드 p48 '전이학습' 방식: 학습된 특징 추출부는 그대로 쓰고,
              출력 부류 수에 맞게 바꾼 후반부만 학습)

사용 예:
    python scripts/train.py --mode full     --data datasets/wildlife3/wildlife3.yaml
    python scripts/train.py --mode freeze10 --data datasets/wildlife3/wildlife3.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "freeze10"], default="full")
    ap.add_argument("--data", default="datasets/wildlife3/wildlife3.yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="COCO 사전학습 가중치")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="GPU가 있으면 0")
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    model = YOLO(args.weights)          # nc=80(COCO) 헤드 -> 데이터의 nc=3에 맞게 자동 교체
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=10 if args.mode == "freeze10" else None,
        close_mosaic=5,                 # 마지막 5 epoch는 모자이크 증강 끄고 원본 분포로 마무리
        cache=True,                     # 이미지를 메모리에 올려 CPU 학습 속도 개선
        seed=0,
        deterministic=True,
        project=str(Path(args.project).resolve()),   # 상대 경로면 Ultralytics가 runs/detect 아래로 옮기므로
        name=f"ft_{args.mode}",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
