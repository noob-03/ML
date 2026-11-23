"""학습한 모델로 영상에서 객체를 탐지하고 추적(ID 부여)한 결과 영상을 만든다.

- 프레임마다 model.track(..., persist=True) 를 호출해 이전 프레임의 추적 상태를 이어 간다.
- 추적기: ByteTrack (Ultralytics 내장 bytetrack.yaml)
- 부류 구분 없는 NMS(agnostic NMS)를 기본으로 쓴다. 기본 NMS는 부류별로 따로 돌아서, 한 동물에
  'rhino' 박스와 'elephant' 박스가 함께 남으면 둘 다 살아남고 추적기가 두 번째 박스에 새 ID를 붙인다.
  이 데이터의 세 동물은 서로 포개질 일이 드물어서, 겹친 박스는 신뢰도가 높은 하나만 남기는 편이 맞다.
  (--no-agnostic-nms 로 끄면 이전 방식)
- 각 ID의 박스 중심점을 프레임마다 모아 궤적 선으로 그리고, 궤적은 CSV로 따로 저장한다.
  (CV-4 슬라이드 p51: '매 프레임 객체 중심점 시각화, trajectory 별도 저장')
- --snapshots 로 지정한 시각의 프레임 4장을 2x2 그림 하나로 저장한다 (README·보고서 그림).

사용 예:
    python scripts/track.py --weights weights/best.pt --source video/source_pan_5s.mp4 \
        --out video/track_demo_5s.mp4 --csv results/trajectories.csv \
        --snapshot-out results/tracking_frames.jpg
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

from track_stats import summarize


def snapshot_grid(frames, out_path, cols=2, width=640, gap=10):
    """(시각, 프레임) 목록을 시각 표시를 넣어 격자 그림 하나로 합친다."""
    tiles = []
    for t, img in frames:
        h = round(img.shape[0] * width / img.shape[1])
        tile = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        label = f"t={t:.2f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(tile, (6, 6), (6 + tw + 14, 6 + th + 16), (40, 40, 40), -1)
        cv2.putText(tile, label, (13, 6 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    th, tw = tiles[0].shape[:2]
    grid = np.full((len(rows) * th + (len(rows) - 1) * gap, cols * tw + (cols - 1) * gap, 3), 255, np.uint8)
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            grid[r * (th + gap): r * (th + gap) + th, c * (tw + gap): c * (tw + gap) + tw] = tile
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(weights, source, out, csv_path, imgsz=416, conf=0.25, tracker="bytetrack.yaml", agnostic_nms=True,
        trail=20, snapshots=(), snapshot_out=None, verbose=True):
    """영상 하나를 탐지·추적해 결과 영상(out, None이면 저장 안 함)과 궤적 CSV를 만들고 ID 통계를 돌려준다."""
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    snap_frames = {round(t * fps): t for t in snapshots}

    writer = raw = None
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_name(out.stem + "_raw.mp4")
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history = defaultdict(list)          # track_id -> [(x, y), ...]
    rows = []                            # CSV 행
    snaps = []
    frame_idx = 0
    nms = "agnostic NMS" if agnostic_nms else "class-wise NMS"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=tracker, imgsz=imgsz,
                          conf=conf, agnostic_nms=agnostic_nms, verbose=False)[0]
        canvas = res.plot(line_width=2)                     # 박스 + 'id:N 부류 신뢰도'
        if res.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cls, cf in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                                       res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                history[tid].append((int(cx), int(cy)))
                rows.append([frame_idx, round(frame_idx / fps, 3), tid, model.names[cls], round(cf, 3),
                             round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                pts = history[tid][-trail:]                 # 현재 보이는 ID만 최근 궤적(중심점 경로)을 그림
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, colors(cls, True), 3, cv2.LINE_AA)
                cv2.circle(canvas, pts[-1], 4, colors(cls, True), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"YOLOv8n fine-tuned (elephant/rhino/zebra) + ByteTrack, {nms}  frame {frame_idx:03d}",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if writer:
            writer.write(canvas)
        if frame_idx in snap_frames:
            snaps.append((snap_frames[frame_idx], canvas.copy()))
        frame_idx += 1
    cap.release()

    if writer:
        writer.release()
        # 브라우저·GitHub에서 바로 재생되도록 H.264로 다시 인코딩
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           check=True)
            raw.unlink()
        else:
            raw.rename(out)

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["frame", "time_s", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        wr.writerows(rows)

    if snapshot_out and snaps:
        snapshot_grid(snaps, snapshot_out)
        if verbose:
            print("그림:", snapshot_out)
    if verbose:
        print(f"{out}: {frame_idx} frames ({frame_idx / fps:.2f}s), imgsz {imgsz}, {nms}")
    return summarize(csv_path, width=w, height=h, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="video/track_demo_5s.mp4")
    ap.add_argument("--csv", default="results/trajectories.csv")
    ap.add_argument("--imgsz", type=int, default=416,
                    help="추론 입력 크기. 학습은 320이지만 1280x720 영상에서는 416일 때 동물 크기가 학습 때와 비슷하다")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=True,
                    help="부류 구분 없이 겹친 박스를 하나만 남김 (기본: 켬)")
    ap.add_argument("--trail", type=int, default=20, help="궤적으로 그릴 최근 프레임 수")
    ap.add_argument("--snapshots", type=float, nargs="*", default=[0.3, 1.5, 2.8, 4.7],
                    help="그림으로 저장할 시각(초)")
    ap.add_argument("--snapshot-out", default=None, help="예: results/tracking_frames.jpg (주지 않으면 저장 안 함)")
    args = ap.parse_args()
    run(args.weights, args.source, args.out, args.csv, args.imgsz, args.conf, args.tracker, args.agnostic_nms,
        args.trail, args.snapshots, args.snapshot_out)


if __name__ == "__main__":
    main()
