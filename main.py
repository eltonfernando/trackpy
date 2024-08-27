# -*- coding: utf-8 -*-
from logging import basicConfig
from trackpy import MultiObjectTracker, DataTracker, Detection, BoundboxDetector, ObjectTracked, BoundboxDTO

basicConfig(level="DEBUG")

track = MultiObjectTracker(DataTracker(fps := 10, max_frame_skipped=10, dist_threshold=50))

for i in range(0, 100):
    box = BoundboxDTO(0.5, 0.5, 0.5, 0.5)
    box_detected = BoundboxDetector(boundbox=box, score=0.5, label="test")
    detection = Detection()
    detection.add_box(box_detected)
    track.update(detection)
