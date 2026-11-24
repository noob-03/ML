# 머신러닝 과제

| 폴더 | 내용 |
| --- | --- |
| [`yolo/`](yolo/) | 과제 2. YOLOv8n을 코끼리·코뿔소·얼룩말 3개 부류로 fine-tuning 하고 5초 탐지·추적 영상 만들기 |

과제 2에서 바로 볼 파일:

- 학습 모델: [`yolo/weights/best.pt`](yolo/weights/best.pt) (최종, 백본 고정), [`yolo/weights/best_full.pt`](yolo/weights/best_full.pt) (비교용, 전체 층 학습)
- 탐지·추적 영상: [`yolo/video/track_demo_5s.mp4`](yolo/video/track_demo_5s.mp4) (5초, 1280×720)
- 학습 과정·성능·재현 방법: [`yolo/README.md`](yolo/README.md)
