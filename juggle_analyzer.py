import collections
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class BallObservation:
    label: str
    center: Tuple[int, int]
    radius: int
    color: Tuple[int, int, int]


class BallTracker:
    def __init__(self, label: str, lower_hsv: Tuple[int, int, int], upper_hsv: Tuple[int, int, int], draw_color: Tuple[int, int, int]):
        self.label = label
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.draw_color = draw_color

    def detect(self, frame: np.ndarray) -> Optional[BallObservation]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 200:
            return None

        ((x, y), radius) = cv2.minEnclosingCircle(best)
        if radius < 8:
            return None

        return BallObservation(self.label, (int(x), int(y)), int(radius), self.draw_color)

    def draw(self, frame: np.ndarray, observation: BallObservation) -> None:
        if observation is None:
            return
        cv2.circle(frame, observation.center, observation.radius, self.draw_color, 2)
        cv2.putText(frame, self.label, (observation.center[0] - 20, observation.center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.draw_color, 2)


class TrajectoryAnalyzer:
    def __init__(self, max_history: int = 60):
        self.history: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=max_history))

    def update(self, observations: List[BallObservation]) -> Dict[str, Dict[str, float]]:
        analysis = {}
        for obs in observations:
            self.history[obs.label].append(obs.center)

        for label, track in self.history.items():
            analysis[label] = {
                'throw_height_px': self._estimate_height(track),
                'speed_px': self._estimate_speed(track),
                'ready': len(track) >= 2,
            }

        analysis['collision_risk'] = self._estimate_collision_risk()
        return analysis

    def _estimate_height(self, track: collections.deque) -> float:
        if len(track) < 3:
            return 0.0
        ys = [p[1] for p in track]
        return float(max(ys) - min(ys))

    def _estimate_speed(self, track: collections.deque) -> float:
        if len(track) < 2:
            return 0.0
        dx = track[-1][0] - track[-2][0]
        dy = track[-1][1] - track[-2][1]
        return float(np.hypot(dx, dy))

    def _estimate_collision_risk(self) -> float:
        labels = list(self.history.keys())
        if len(labels) < 2:
            return 0.0

        min_distance = float('inf')
        for i in range(len(labels)):
            if len(self.history[labels[i]]) < 2:
                continue
            for j in range(i + 1, len(labels)):
                if len(self.history[labels[j]]) < 2:
                    continue
                a = np.array(self.history[labels[i]][-1], dtype=np.float32)
                b = np.array(self.history[labels[j]][-1], dtype=np.float32)
                dist = np.linalg.norm(a - b)
                min_distance = min(min_distance, dist)

        if min_distance == float('inf'):
            return 0.0
        return float(max(0.0, min(1.0, (120.0 - min_distance) / 120.0)))


class HandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.6)
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=40))

    def analyze(self, frame: np.ndarray) -> Dict[str, str]:
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        summary = {
            'hands_detected': 0,
            'pattern': 'Unknown',
        }
        if not results.multi_hand_landmarks:
            return summary

        summary['hands_detected'] = len(results.multi_hand_landmarks)
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            self.history[f'hand{hand_index}'].append((wrist.x, wrist.y))

        summary['pattern'] = self._estimate_pattern()
        return summary

    def _estimate_pattern(self) -> str:
        if len(self.history) < 2:
            return 'Waiting for motion data'

        patterns = []
        for key, track in self.history.items():
            if len(track) < 5:
                continue
            xs = [p[0] for p in track]
            if xs[-1] > xs[0]:
                patterns.append('moving right')
            else:
                patterns.append('moving left')

        if len(patterns) == 2:
            if patterns[0] != patterns[1]:
                return 'Alternating rhythm'
            return 'Synchronized pattern'
        return patterns[0] if patterns else 'Stable or paused'


def create_default_trackers() -> List[BallTracker]:
    return [
        BallTracker('red', (0, 120, 70), (10, 255, 255), (0, 0, 255)),
        BallTracker('yellow', (20, 120, 70), (35, 255, 255), (0, 255, 255)),
        BallTracker('blue', (90, 100, 70), (140, 255, 255), (255, 0, 0)),
    ]


def draw_overlay(frame: np.ndarray, analysis: Dict[str, Dict[str, float]], hand_data: Dict[str, str]) -> None:
    y = 25
    for label, values in analysis.items():
        if label == 'collision_risk':
            text = f'Collision risk: {values:.2f}'
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30
            continue

        text = f'{label}: height={values["throw_height_px"]:.0f}px speed={values["speed_px"]:.1f}'
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    cv2.putText(frame, f'Hands: {hand_data["hands_detected"]}', (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 25
    cv2.putText(frame, f'Pattern: {hand_data["pattern"]}', (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    trackers = create_default_trackers()
    trajectory = TrajectoryAnalyzer()
    hand_analyzer = HandAnalyzer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open webcam. Please check your camera and try again.')
        return

    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        observations = []
        for tracker in trackers:
            obs = tracker.detect(frame)
            if obs:
                observations.append(obs)
                tracker.draw(frame, obs)

        analysis = trajectory.update(observations)
        hand_data = hand_analyzer.analyze(frame)
        draw_overlay(frame, analysis, hand_data)

        fps = 1.0 / (time.time() - last_time) if last_time else 0.0
        last_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.1f}', (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('JugglePro - Press q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
