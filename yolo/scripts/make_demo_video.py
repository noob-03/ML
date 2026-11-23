"""테스트 이미지 3장으로 5초짜리 '카메라 패닝' 영상을 만든다 (추적 시연용 입력 영상).

실제 촬영 영상을 구할 수 없는 환경이라, 학습에 쓰지 않은 test 이미지를 가로로 이어 붙인 뒤
1280x720 창을 왼쪽에서 오른쪽으로 일정한 속도로 움직여 30fps 영상으로 만든다.
화면 안에서 동물이 들어오고 지나가므로 객체 추적(ID 유지)을 확인할 수 있다.
실제 영상이 있으면 이 단계는 건너뛰고 track.py 에 그 영상을 바로 넣으면 된다.

사용 예:
    python scripts/make_demo_video.py --images "datasets/wildlife3/images/test/2 (227).jpg" \
        "datasets/wildlife3/images/test/3 (209).jpg" "datasets/wildlife3/images/test/4 (213).jpg" \
        --out video/source_pan_5s.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def build_panorama(paths, height, blend):
    """이미지들을 같은 높이로 맞추고, 이음새는 blend 픽셀 폭으로 부드럽게 섞어 가로로 잇는다."""
    pano = None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        w = round(img.shape[1] * height / img.shape[0])
        img = cv2.resize(img, (w, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        if pano is None:
            pano = img
            continue
        alpha = np.linspace(0, 1, blend, dtype=np.float32)[None, :, None]
        seam = pano[:, -blend:] * (1 - alpha) + img[:, :blend] * alpha
        pano = np.concatenate([pano[:, :-blend], seam, img[:, blend:]], axis=1)
    return pano.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="video/source_pan_5s.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--blend", type=int, default=120, help="이음새를 섞는 폭(픽셀)")
    args = ap.parse_args()

    pano = build_panorama(args.images, args.height, args.blend)
    n = int(round(args.fps * args.seconds))
    travel = pano.shape[1] - args.width
    if travel <= 0:
        raise ValueError("이어 붙인 이미지가 화면 폭보다 좁습니다. 이미지를 더 넣으세요.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    for i in range(n):
        x = round(travel * i / (n - 1))           # 등속 패닝
        writer.write(pano[:, x : x + args.width])
    writer.release()
    if shutil.which("ffmpeg"):                    # 브라우저·GitHub에서 재생되도록 H.264로 변환
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)
    print(f"{out}: {n} frames, {n / args.fps:.2f}s, panorama {pano.shape[1]}x{pano.shape[0]}, {travel / (n - 1):.1f}px/frame")


if __name__ == "__main__":
    main()
