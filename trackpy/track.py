# -*- coding: utf-8 -*-

from collections import deque
from typing import List, Tuple
from logging import getLogger
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import KalmanBox
from .core import Detection, BoundboxDetector


class DataTracker:
    def __init__(self, fps: int = 10, max_frame_skipped: int = 10, dist_threshold: int = 50) -> None:
        self.log = getLogger("trackpy")
        self.fps = fps
        self.max_frame_skipped = max_frame_skipped
        self.dist_threshold = dist_threshold

    def get_fps(self) -> int:
        return self.fps


class ObjectTracked:
    def __init__(self, box_detection: BoundboxDetector, tracker_id: int) -> None:
        self.log = getLogger("trackpy")
        self.tracker_id = tracker_id
        self.boundbox_detection = box_detection
        self.raster = None

        self.velocity = 0
        self.__skipped_frames = 0
        self.trace = deque(maxlen=120)  # index 0 removido se len >30
        self.confidence = 0.4
        # self.update(boundbox, score=score, label=label)
        self.kalman_filter = KalmanBox(box_detection.to_rect())
        self.data_activated_counter = {}
        self.count_timer_counted = 0

    def get_label(self) -> str:
        return self.boundbox_detection.get_label()

    def len_trace(self):
        return len(self.trace)

    def get_skipped_frames(self) -> None:
        return self.__skipped_frames

    def reset_skipped_frames(self) -> None:
        self.__skipped_frames = 0

    def increment_skipped_frames(self) -> None:
        self.__skipped_frames += 1

    def is_moving(self):
        return self.get_velocity() > 30 and self.confidence > 0.7

    def _calcule_prob_bayes(self, prob_apriore, prob_posteriore):
        """
        :param prob_apriore: confiança anterior
        :param prob_posteriore: nova confiança
        :return: prob
        """
        prob = (prob_posteriore * prob_apriore) / (prob_posteriore * prob_apriore + (1 - prob_posteriore) * (1 - prob_apriore))
        prob = min(prob, 0.99)
        prob = max(prob, 0.1)

        return prob

    def update_score(self, score: float) -> None:
        score = max(self._mover_confidence(), score)
        prob = self._calcule_prob_bayes(self.confidence, score)
        self.confidence = prob

    def _mover_confidence(self):
        return 0.18
        # self.confidence = self._calcule_prob_bayes(self.confidence, erro)

    def is_counter(self, key: str):
        if not key in self.data_activated_counter:
            return False
        return self.data_activated_counter[key]

    def get_confidencia(self):
        return self.confidence

    def set_counter(self, key: str, value: bool) -> None:
        self.data_activated_counter[key] = value

    def update(self, box_detection: BoundboxDetector) -> None:
        self.update_score(box_detection.get_score())
        self.update_bounndbox(box_detection)

    def get_velocity(self) -> float:
        return self.velocity

    def get_xmin(self) -> int:
        return int(self.boundbox_detection.x_min)

    def get_ymin(self) -> int:
        return int(self.boundbox_detection.y_min)

    def get_xmax(self) -> int:
        return int(self.boundbox_detection.x_max)

    def get_ymax(self) -> int:
        return int(self.boundbox_detection.y_max)

    def get_center_tuple(self) -> Tuple[int, int]:
        return self.boundbox_detection.to_center_tuple()

    def get_center(self) -> np.ndarray:
        center_x, center_y = self.get_center_tuple()
        return np.array([center_x, center_y], dtype=np.float32)

    def update_bounndbox(self, bndbox: BoundboxDetector) -> None:
        if bndbox.is_box_empty():
            (
                x_min,
                x_velocity,
                y_min,
                y_velocity,
                width,
                hieght,
            ) = self.kalman_filter.update(None)

            self.velocity = np.linalg.norm(np.array([x_velocity, y_velocity]))
            self.boundbox_detection.set_rect(x_min, y_min, width, hieght)
            self.trace.append(self.get_center().astype(np.int32))
        else:
            x, y, w, h = bndbox.to_rect()
            (
                x_min,
                x_velocity,
                y_min,
                y_velocity,
                width,
                hieght,
            ) = self.kalman_filter.update(np.array([[x], [y], [w], [h]]))
            self.velocity = np.linalg.norm(np.array([x_velocity, y_velocity]))
            self.boundbox_detection.set_rect(x_min, y_min, width, hieght)
            self.trace.append(self.get_center().astype(np.int32))


class MultiObjectTracker:
    def __init__(self, data: DataTracker):
        self.log = getLogger("trackpy")

        self.fps = data.fps
        self.max_frame_skipped = data.max_frame_skipped
        self.dist_threshold = data.dist_threshold
        self.historic: List[ObjectTracked] = []
        self.track_id = 0
        self.color_line = (0, 0, 255)

        self.assignment = []

        self.track_colors = [
            (255, 8, 127),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 127, 255),
            (127, 0, 255),
            (127, 0, 127),
        ]

    def update(self, object_detected: Detection):
        self.log.debug(f"update {object_detected}")
        # add sequencia de pontos incial
        if len(self.historic) == 0:
            self.track_id = 0

        # nada detectado atualiza tracker
        if not object_detected.status:
            self.assignment = [-1] * len(self.historic)
            self.remove_object_confidencia_low()
            self.update_list_object(object_detected)
            return

        detections = np.asarray(object_detected.get_center_box()).astype(np.int32)

        if len(self.historic) == 0:
            self._add_point_start(detections, object_detected)
            return

        cost = self._sincronize_points(detections)

        # remove se distancia muito grande
        for i, value in enumerate(self.assignment):
            if value != -1:
                if cost[i][value] > self.dist_threshold:
                    self.assignment[i] = -1
            else:
                self.historic[i].increment_skipped_frames()

        self.remove_object_confidencia_low()

        # Add novos pontos

        for i in range(len(detections)):
            if i not in self.assignment:
                box_detected = object_detected.get_box(i)
                self.insert_objeto_detected(box_detected)

        self.update_list_object(object_detected)

    def remove_object_confidencia_low(self):
        """_summary_

        remove objetos com confiança abaixo de 0.2 e tem mais de 5 pontos
        também remove se o objeto tiver max_frame_skipped sem detectar
        """
        del_tracks = []
        for i, obj in enumerate(self.historic):
            if obj.get_confidencia() < 0.2 and len(obj.trace) > 5 or self.historic[i].get_skipped_frames() > self.max_frame_skipped:
                del_tracks.append(i)

        if len(del_tracks) != 0:
            del_tracks = sorted(del_tracks, key=int, reverse=True)
            for my_id in del_tracks:
                del self.historic[my_id]
                del self.assignment[my_id]

    def _add_point_start(self, detections: np.ndarray, object_detected: Detection):
        for i in range(len(detections)):
            box_detected = object_detected.get_box(i)
            self.insert_objeto_detected(box_detected)

    def insert_objeto_detected(self, box_detected: BoundboxDetector):
        obj = ObjectTracked(
            box_detection=box_detected,
            tracker_id=self.track_id,
        )
        self.track_id += 1
        self.historic.append(obj)

    def _sincronize_points(self, detections):
        size_historic = len(self.historic)

        cost = []
        for i in range(size_historic):
            diff = np.linalg.norm(self.historic[i].get_center() - detections.reshape(-1, 2), axis=1)
            cost.append(diff)
        cost = np.array(cost) * 0.4
        row, col = linear_sum_assignment(cost)

        self.assignment = [-1] * size_historic

        for i, _ in enumerate(row):
            self.assignment[row[i]] = col[i]
        return cost

    def update_list_object(self, object_detected: Detection):

        if len(self.assignment) == 0:
            return
        for i, value in enumerate(self.assignment):
            if value != -1:
                self.historic[i].reset_skipped_frames()
                box_detector = object_detected.get_box(value)
                self.historic[i].update(box_detector)

            else:
                self.historic[i].increment_skipped_frames()
                default_detect = BoundboxDetector(boundbox=BoundboxDTO(xmin=0, ymin=0, xmax=0, ymax=0), score=0, label="")
                self.historic[i].update(default_detect)

    def draw(self, img: np.ndarray) -> None:
        """
        Desenha animação das pessoas detectadas
        :param img: parametro passado por referencia
        :return:
        """

        for obj in self.historic:
            if len(obj.trace) > 0:
                x_min = obj.get_xmin()
                y_min = obj.get_ymin()
                x_max = obj.get_xmax()
                y_max = obj.get_ymax()

                for j in range(len(obj.trace) - 1):
                    point_x = int(obj.trace[j][0])
                    point_y = int(obj.trace[j][1])
                    next_point_x = int(obj.trace[j + 1][0])
                    next_point_y = int(obj.trace[j + 1][1])
                    clr = obj.tracker_id % 9
                    cv2.line(
                        img,
                        (point_x, point_y),
                        (next_point_x, next_point_y),
                        self.track_colors[clr],
                        2,
                        cv2.LINE_AA,
                    )

                color_box = (225, 0, 0)
                for i, (key, value) in enumerate(obj.data_activated_counter.items()):
                    if value:
                        color_box = self.track_colors[i % 9]
                        cv2.putText(
                            img,
                            f"{key}: {value}",
                            (x_max + 2, y_min + 20 * (i + 1)),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.55,
                            (255, 0, 225),
                        )

                cv2.rectangle(
                    img,
                    (x_min, y_min),
                    (x_max, y_min + 12),
                    color_box,
                    cv2.FILLED,
                )
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_box, 1)
                cv2.rectangle(
                    img,
                    (x_min, y_max),
                    (x_max, y_max - 12),
                    color_box,
                    cv2.FILLED,
                )

                confi = round(obj.get_confidencia() * 100, 1)

                cv2.putText(
                    img,
                    str(confi) + "%",
                    (x_min + 2, y_max - 3),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.25,
                    (255, 255, 225),
                )
                cv2.putText(
                    img,
                    "ID " + str(obj.tracker_id),
                    (x_min, y_min + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                )
