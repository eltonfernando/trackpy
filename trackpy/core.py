# -*- coding: utf-8 -*-
from typing import List, Tuple
import logging

from cv2 import rectangle
from numpy import ndarray


class BoundboxDetector:
    def __init__(self, x_min: int = 0, y_min: int = 0, x_max: int = 0, y_max: int = 0, score: float = 0, label: str = ""):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.score = score
        self.label = label

    def is_box_empty(self) -> bool:
        return self.x_min == 0 and self.y_min == 0 and self.x_max == 0 and self.y_max == 0

    def to_box(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_rect(self) -> List[int]:
        return [self.x_min, self.y_min, (self.x_max - self.x_min), (self.y_max - self.y_min)]

    def to_center_tuple(self) -> Tuple[int, int]:
        return (int((self.x_max + self.x_min) / 2), int((self.y_max + self.y_min) / 2))

    def set_rect(self, x_min: int, y_min: int, width: int, hieght: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_min + width
        self.y_max = y_min + hieght

    def get_score(self) -> float:
        return self.score

    def get_label(self) -> str:
        return self.label

    def compute_inscrito_in(self, x_min, y_min, x_max, y_max):
        if self.x_min < x_min:
            return False
        if self.y_min < y_min:
            return False
        if self.x_max > x_max:
            return False
        if self.y_max > y_max:
            return False
        return True

    def __repr__(self):
        return f"BoundboxDetector(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max}, score={self.score}, label={self.label})"


class Detection:
    def __init__(self) -> None:
        self.log = logging.getLogger("detection")
        self.status = False
        self.list_box_detector: List[BoundboxDetector] = []
        self.min_score = 1.0

    def add_box(self, box_detector: BoundboxDetector) -> None:
        self.status = True
        if box_detector.score < self.min_score:
            self.min_score = box_detector.score
        self.list_box_detector.append(box_detector)

    def get_box(self, index: int) -> BoundboxDetector:
        return self.list_box_detector[index]

    def map_move_to(self, offset_x: int, offset_y: int) -> None:
        for box_detector in self.list_box_detector:
            box_detector.x_min += offset_x
            box_detector.y_min += offset_y
            box_detector.x_max += offset_x
            box_detector.y_max += offset_y

    def get_min_score(self) -> float:
        return self.min_score

    def __repr__(self):
        return f"Detection(box={self.list_box_detector},status {self.status})"

    def get_center_box(self) -> List[float]:
        center_box = []
        if self.status:
            for box in self.list_box_detector:
                center_box.append([(box.x_max + box.x_min) / 2, (box.y_max + box.y_min) / 2])
        return center_box

    def compute_dist_threshold(self):
        self.log.debug(f"computing dist threshold")
        size = []
        compute_sum = 0
        if self.status:
            for box in self.list_box_detector:
                w, h = box.x_max - box.x_min, box.y_max - box.y_min
                raio = (w + h) / 2
                compute_sum += raio
                size.append(raio)
        return int(compute_sum / len(size) * 0.4)

    def draw(self, frame: ndarray):
        for box in self.list_box_detector:
            rectangle(frame, (int(box.x_min), int(box.y_min)), (int(box.x_max), int(box.y_max)), (0, 255, 0), 2)
