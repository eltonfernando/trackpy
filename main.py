# -*- coding: utf-8 -*-
from logging import basicConfig
from trackpy import MultiObjectTracker, DataTracker, Detection, BoundboxDetector, ObjectTracked

basicConfig(level="DEBUG")

track = MultiObjectTracker(DataTracker(fps := 10, max_frame_skipped=10, dist_threshold=50))

for i in range(0, 100):
    box_detected = BoundboxDetector(x_min=0, y_min=100, x_max=100, y_max=400, score=0.5, label="test")
    detection = Detection()
    detection.add_box(box_detected)
    track.update(detection)
    print(track.historic)
