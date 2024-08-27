# -*- coding: utf-8 -*-
from typing import List, Tuple
import logging

from cv2 import rectangle
from numpy import ndarray


class BoundboxDTO:
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def empty(self) -> bool:
        return self.x_min == 0 and self.y_min == 0 and self.x_max == 0 and self.y_max == 0

    def to_box(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_rect(self) -> List[int]:
        return [self.x_min, self.y_min, (self.x_max - self.x_min), (self.y_max - self.y_min)]

    def to_center_tuple(self) -> Tuple[int, int]:
        return (int((self.x_max + self.x_min) / 2), int((self.y_max + self.y_min) / 2))

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    def __repr__(self):
        return f"BoundboxDTO(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"


class BoundboxDetector:
    def __init__(self, boundbox: BoundboxDTO, score: float, label: str):
        self.box = boundbox
        self.score = score
        self.label = label

    def get_boundbox(self) -> BoundboxDTO:
        return self.box

    def set_rect(self, x_min: int, y_min: int, width: int, hieght: int):
        self.box = BoundboxDTO(x_min, y_min, x_min + width, y_min + hieght)

    def to_rect(self) -> BoundboxDTO:
        return self.box.to_rect()

    def get_score(self) -> float:
        return self.score

    def get_label(self) -> str:
        return self.label

    def compute_inscrito_in(self, box: BoundboxDTO):
        if self.box.x_min < box.x_min:
            return False
        if self.box.y_min < box.y_min:
            return False
        if self.box.x_max > box.x_max:
            return False
        if self.box.y_max > box.y_max:
            return False
        return True

    def __repr__(self):
        return f"BoundboxDetector(box={self.box}, score={self.score}, label={self.label})"


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
            box = box_detector.box
            box.x_min += offset_x
            box.y_min += offset_y
            box.x_max += offset_x
            box.y_max += offset_y

    def get_min_score(self) -> float:
        return self.min_score

    def __repr__(self):
        return f"Detection(box={self.list_box_detector},status {self.status})"

    def get_center_box(self) -> List[float]:
        center_box = []
        if self.status:
            for box_detector in self.list_box_detector:
                box = box_detector.box
                center_box.append([(box.x_max + box.x_min) / 2, (box.y_max + box.y_min) / 2])
        return center_box

    def compute_dist_threshold(self):
        self.log.debug(f"computing dist threshold")
        size = []
        compute_sum = 0
        if self.status:
            for box_detector in self.list_box_detector:
                box = box_detector.box
                w, h = box.x_max - box.x_min, box.y_max - box.y_min
                raio = (w + h) / 2
                compute_sum += raio
                size.append(raio)
        return int(compute_sum / len(size) * 0.4)

    def draw(self, frame: ndarray):
        for box_detector in self.list_box_detector:
            box = box_detector.box
            rectangle(frame, (int(box.x_min), int(box.y_min)), (int(box.x_max), int(box.y_max)), (0, 255, 0), 2)
