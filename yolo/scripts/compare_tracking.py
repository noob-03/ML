"""추론 입력 크기와 NMS 방식에 따라 같은 영상의 추적 결과가 어떻게 달라지는지 비교한다 (보고서 4장 표).

조합마다 track.py 와 같은 방식으로 추적하고(영상은 저장하지 않음), track_stats.py 기준으로
전체 ID 수, 10프레임 이상 이어진 ID 수, 짧게 끊긴 ID 수(원인별)를 CSV 한 장으로 모은다.

사용 예:
    python scripts/compare_tracking.py --weights weights/best.pt --source video/source_pan_5s.mp4 \
        --out results/tracking_settings.csv
"""

import argparse
import csv
import tempfile
from collections import Counter
from pathlib import Path

from track import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", default="video/source_pan_5s.mp4")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--out", default="results/tracking_settings.csv")
    args = ap.parse_args()

    table = []
    with tempfile.TemporaryDirectory() as tmp:
        for sz in args.imgsz:
            for agnostic in (False, True):
                s = run(args.weights, args.source, None, Path(tmp) / "t.csv", imgsz=sz, agnostic_nms=agnostic,
                        verbose=False)
                causes = Counter(s["short_causes"].values())
                table.append([sz, "agnostic" if agnostic else "class-wise", s["ids"], len(s["long_ids"]),
                              len(s["short_ids"]), causes["duplicate"], causes["edge"], causes["other"]])
                print(table[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["imgsz", "nms", "track_ids", "ids_10plus_frames", "ids_under_10_frames",
                     "short_duplicate", "short_edge", "short_other"])
        wr.writerows(table)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
