import argparse
import math
import time
from collections import deque

import cv2
import numpy as np


class BallTrack:
    def __init__(self, track_id, initial_position, initial_radius):
        self.track_id = track_id
        self.radius = initial_radius
        self.path = deque(maxlen=40)
        self.path.append(initial_position)
        self.last_seen = time.time()
        self.misses = 0
        self.max_missed = 10

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array(
            [initial_position[0], initial_position[1], 0.0, 0.0], np.float32
        )

    def predict(self):
        prediction = self.kalman.predict()
        return float(prediction[0]), float(prediction[1])

    def update(self, position, radius):
        measurement = np.array([np.float32(position[0]), np.float32(position[1])])
        self.kalman.correct(measurement)
        self.path.append(position)
        self.radius = radius
        self.last_seen = time.time()
        self.misses = 0

    def mark_missed(self):
        self.misses += 1

    def is_lost(self):
        return self.misses > self.max_missed


class BallTracker:
    def __init__(self, max_balls=5, max_distance=100, debug=False):
        self.max_balls = max_balls
        self.max_distance = max_distance
        self.debug = debug
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        if len(detections) == 0:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [track for track in self.tracks if not track.is_lost()]
            return

        predictions = [track.predict() for track in self.tracks]
        assigned_tracks = [-1] * len(self.tracks)
        assigned_detections = [-1] * len(detections)

        for track_index, prediction in enumerate(predictions):
            best_distance = float("inf")
            best_det = -1
            for det_index, det in enumerate(detections):
                if assigned_detections[det_index] != -1:
                    continue
                distance = math.dist(prediction, (det[0], det[1]))
                if distance < best_distance:
                    best_distance = distance
                    best_det = det_index
            if best_det != -1 and best_distance <= self.max_distance:
                assigned_tracks[track_index] = best_det
                assigned_detections[best_det] = track_index

        for track_index, track in enumerate(self.tracks):
            det_index = assigned_tracks[track_index]
            if det_index != -1:
                det = detections[det_index]
                track.update((det[0], det[1]), det[2])
            else:
                track.mark_missed()

        for det_index, det in enumerate(detections):
            if assigned_detections[det_index] == -1 and len(self.tracks) < self.max_balls:
                self._create_track(det)

        self.tracks = [track for track in self.tracks if not track.is_lost()]

    def _create_track(self, detection):
        self.tracks.append(
            BallTrack(self.next_id, (detection[0], detection[1]), detection[2])
        )
        self.next_id += 1

    def draw(self, frame, yellow_mask=None):
        for track in self.tracks:
            if len(track.path) == 0:
                continue

            current_position = track.path[-1]
            predicted_position = track.predict()
            center = (int(current_position[0]), int(current_position[1]))
            predicted_center = (int(predicted_position[0]), int(predicted_position[1]))

            cv2.circle(frame, center, int(track.radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 255, 0), -1)
            cv2.line(frame, center, predicted_center, (255, 0, 0), 1)
            cv2.putText(
                frame,
                f"#{track.track_id}",
                (center[0] - 10, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            for i in range(1, len(track.path)):
                p0 = (int(track.path[i - 1][0]), int(track.path[i - 1][1]))
                p1 = (int(track.path[i][0]), int(track.path[i][1]))
                cv2.line(frame, p0, p1, (0, 180, 255), 2)

            if yellow_mask is not None:
                x, y = center
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    color_score = int(yellow_mask[y, x])
                    status = "Y" if color_score > 0 else "?"
                    cv2.putText(
                        frame,
                        status,
                        (center[0] + 10, center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255) if status == "Y" else (0, 0, 255),
                        2,
                    )

    def get_stats(self):
        return len(self.tracks)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple multi-ball circle tracker")
    parser.add_argument("--flip", action="store_true", help="Flip the camera horizontally for a mirror-free view")
    parser.add_argument("--max-balls", type=int, default=5, help="Maximum number of simultaneous ball tracks")
    parser.add_argument("--max-distance", type=int, default=100, help="Maximum assignment distance in pixels")
    parser.add_argument("--min-radius", type=int, default=20, help="Minimum circle radius to detect")
    parser.add_argument("--max-radius", type=int, default=70, help="Maximum circle radius to detect")
    parser.add_argument("--yellow-only", action="store_true", help="Validate detected circles with yellow color when enabled")
    parser.add_argument("--debug", action="store_true", help="Show debug overlays and printed tracker stats")
    return parser.parse_args()


def get_yellow_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 80, 120], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.medianBlur(mask, 5)


def detect_round_objects(frame, min_radius, max_radius, use_color=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=70,
        param1=100,
        param2=34,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    mask = get_yellow_mask(frame) if use_color else None
    accepted = []
    for x, y, r in np.round(circles[0]).astype(int):
        if r < min_radius or r > max_radius:
            continue

        if mask is not None:
            x0 = max(0, x - r)
            y0 = max(0, y - r)
            x1 = min(mask.shape[1], x + r)
            y1 = min(mask.shape[0], y + r)
            region = mask[y0:y1, x0:x1]
            if region.size == 0 or np.count_nonzero(region) < 0.18 * region.size:
                continue

        accepted.append((float(x), float(y), float(r)))
        if len(accepted) >= 5:
            break

    return accepted


def detect_round_objects_roi(frame, center, search_radius, min_radius, max_radius, use_color=False):
    h, w = frame.shape[:2]
    x, y = int(center[0]), int(center[1])
    margin = max(80, int(search_radius * 2.0))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + margin)
    y1 = min(h, y + margin)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=24,
        minRadius=max(8, min_radius // 2),
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    mask = get_yellow_mask(roi) if use_color else None
    accepted = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        if r < min_radius or r > max_radius:
            continue

        if mask is not None:
            rx0 = max(0, cx - r)
            ry0 = max(0, cy - r)
            rx1 = min(mask.shape[1], cx + r)
            ry1 = min(mask.shape[0], cy + r)
            region = mask[ry0:ry1, rx0:rx1]
            if region.size == 0 or np.count_nonzero(region) < 0.18 * region.size:
                continue

        accepted.append((float(cx + x0), float(cy + y0), float(r)))
        if len(accepted) >= 3:
            break

    return accepted


def merge_detections(detections, max_count=6):
    unique = []
    for det in detections:
        if all(math.hypot(det[0] - u[0], det[1] - u[1]) > 40 for u in unique):
            unique.append(det)
            if len(unique) >= max_count:
                break
    return unique


def draw_debug_overlay(frame, circles, mask):
    for circle in circles:
        x, y, radius = circle
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    if mask is not None:
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay[:, :, 1:] = 0
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def main():
    args = parse_arguments()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = BallTracker(max_balls=args.max_balls, max_distance=args.max_distance, debug=args.debug)
    last_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        mask = get_yellow_mask(frame) if args.yellow_only else None
        detections = []

        if tracker.tracks:
            for track in tracker.tracks:
                predicted = track.predict()
                detections.extend(
                    detect_round_objects_roi(
                        frame,
                        predicted,
                        max(40, track.radius * 2),
                        args.min_radius,
                        args.max_radius,
                        use_color=args.yellow_only,
                    )
                )

            if frame_count % 4 == 0 or len(detections) == 0:
                detections.extend(detect_round_objects(frame, args.min_radius, args.max_radius, use_color=args.yellow_only))
        else:
            detections = detect_round_objects(frame, args.min_radius, args.max_radius, use_color=args.yellow_only)

        detections = merge_detections(detections, max_count=args.max_balls + 2)

        tracker.update(detections)
        tracker.draw(frame, yellow_mask=mask if args.yellow_only else None)

        if args.debug:
            fps = 1.0 / max(1e-6, time.time() - last_time)
            last_time = time.time()
            cv2.putText(frame, f"Tracks: {tracker.get_stats()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            draw_debug_overlay(frame, detections, mask)

        cv2.imshow("Ball Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
