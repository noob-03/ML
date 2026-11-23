"""track.py가 저장한 궤적 CSV를 요약한다 (보고서 4장의 ID 통계는 이 스크립트로 셌다).

- ID별: 나타난 프레임 수, 처음·마지막 프레임, 부류별 프레임 수, 평균 신뢰도
- 오래 이어진 ID(기본 10프레임 이상)와 짧게 끊긴 ID의 수
- 짧은 ID를 원인별로 나눈다
    duplicate : 같은 프레임에 이미 있는 다른 ID의 박스와 겹친다 (한 동물에 박스 두 개 = 중복 박스)
    edge      : 박스가 화면 가장자리에 닿아 있다 (동물이 화면에 들어오거나 나가는 순간)
    other     : 그 밖의 짧은 검출

사용 예:
    python scripts/track_stats.py --csv results/trajectories.csv
"""

import argparse
import csv
from collections import Counter, defaultdict


def iou_or_cover(a, b):
    """IoU와 '작은 박스가 큰 박스에 덮인 비율' 중 큰 값. 중복 박스는 크기가 달라도 거의 포개진다."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    union = area(a) + area(b) - inter
    return max(inter / union if union else 0.0, inter / min(area(a), area(b)) if min(area(a), area(b)) else 0.0)


def summarize(csv_path, min_frames=10, width=1280, height=720, overlap=0.6, edge=8, verbose=True):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    by_id, by_frame = defaultdict(list), defaultdict(list)
    for r in rows:
        box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        rec = (int(r["frame"]), int(r["track_id"]), r["class"], float(r["conf"]), box)
        by_id[rec[1]].append(rec)
        by_frame[rec[0]].append(rec)

    long_ids = sorted(t for t, recs in by_id.items() if len(recs) >= min_frames)
    short_ids = sorted(t for t, recs in by_id.items() if len(recs) < min_frames)
    causes = {}
    for t in short_ids:
        dup = sum(any(o[1] != t and iou_or_cover(box, o[4]) >= overlap for o in by_frame[f])
                  for f, _, _, _, box in by_id[t])
        at_edge = sum(box[0] <= edge or box[2] >= width - edge or box[1] <= edge or box[3] >= height - edge
                      for _, _, _, _, box in by_id[t])
        n = len(by_id[t])
        causes[t] = "duplicate" if dup * 2 > n else "edge" if at_edge * 2 > n else "other"

    if verbose:
        print(f"{csv_path}: 행 {len(rows)}개, 프레임 {len(by_frame)}개, track ID {len(by_id)}개")
        print(f"{'id':>4} {'frames':>6} {'first':>5} {'last':>5}  {'mean conf':>9}  부류별 프레임 수")
        for t, recs in sorted(by_id.items()):
            cls = Counter(c for _, _, c, _, _ in recs)
            tag = "" if t in long_ids else f"  <- 짧은 ID ({causes[t]})"
            print(f"{t:>4} {len(recs):>6} {recs[0][0]:>5} {recs[-1][0]:>5}  {sum(r[3] for r in recs) / len(recs):>9.2f}"
                  f"  {dict(cls)}{tag}")
        print(f"\n{min_frames}프레임 이상 이어진 ID: {len(long_ids)}개 {long_ids}")
        print(f"{min_frames}프레임 미만으로 끊긴 ID: {len(short_ids)}개, 원인별 {dict(Counter(causes.values()))}")
    return {"ids": len(by_id), "long_ids": long_ids, "short_ids": short_ids, "short_causes": causes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/trajectories.csv")
    ap.add_argument("--min-frames", type=int, default=10, help="이 프레임 수 이상 이어진 ID를 '오래 이어진 ID'로 센다")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    summarize(args.csv, args.min_frames, args.width, args.height)


if __name__ == "__main__":
    main()
