"""학습한 모델로 영상에서 객체를 탐지하고 추적(ID 부여)한 결과 영상을 만든다.

- 프레임마다 model.track(..., persist=True) 를 호출해 이전 프레임의 추적 상태를 이어 간다.
- 추적기: ByteTrack (Ultralytics 내장 bytetrack.yaml)
- 각 ID의 박스 중심점을 프레임마다 모아 궤적 선으로 그리고, 궤적은 CSV로 따로 저장한다.
  (CV-4 슬라이드 p51: '매 프레임 객체 중심점 시각화, trajectory 별도 저장')

사용 예:
    python scripts/track.py --weights weights/best.pt --source video/source_pan_5s.mp4 \
        --out video/track_demo_5s.mp4 --csv results/trajectories.csv
"""

import argparse
import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="video/track_demo_5s.mp4")
    ap.add_argument("--csv", default="results/trajectories.csv")
    ap.add_argument("--imgsz", type=int, default=320, help="학습 때와 같은 입력 크기")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--trail", type=int, default=20, help="궤적으로 그릴 최근 프레임 수")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history = defaultdict(list)          # track_id -> [(x, y), ...]
    rows = []                            # CSV 행
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=args.tracker, imgsz=args.imgsz,
                          conf=args.conf, verbose=False)[0]
        canvas = res.plot(line_width=2)                     # 박스 + 'id:N 부류 신뢰도'
        if res.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                                         res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                history[tid].append((int(cx), int(cy)))
                rows.append([frame_idx, round(frame_idx / fps, 3), tid, model.names[cls], round(conf, 3),
                             round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                pts = history[tid][-args.trail:]            # 현재 보이는 ID만 최근 궤적(중심점 경로)을 그림
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, colors(cls, True), 3, cv2.LINE_AA)
                cv2.circle(canvas, pts[-1], 4, colors(cls, True), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"YOLOv8n fine-tuned (elephant/rhino/zebra) + ByteTrack  frame {frame_idx:03d}",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        frame_idx += 1
    cap.release()
    writer.release()

    # 브라우저·GitHub에서 바로 재생되도록 H.264로 다시 인코딩
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        wr.writerows(rows)

    per_id = defaultdict(lambda: [0, defaultdict(int)])
    for r in rows:
        per_id[r[2]][0] += 1
        per_id[r[2]][1][r[3]] += 1
    print(f"{out}: {frame_idx} frames ({frame_idx / fps:.2f}s), track ID {len(per_id)}개")
    for tid, (n, cls) in sorted(per_id.items()):
        print(f"  id {tid}: {n} frames, {dict(cls)}")


if __name__ == "__main__":
    main()
