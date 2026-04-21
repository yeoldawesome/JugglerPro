import argparse
import math
import time
from collections import deque

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV is required to run this tracker. Install it with: pip install opencv-python numpy"
    ) from exc

import numpy as np


class BallTrack:
    def __init__(self, track_id, initial_position, initial_time):
        self.id = track_id
        self.last_seen = initial_time
        self.predicted_position = np.array(initial_position, dtype=np.float32)
        self.smoothed_position = np.array(initial_position, dtype=np.float32)
        self.history = deque(maxlen=32)
        self.age = 0
        self.missed_frames = 0
        self.color = (int(np.random.randint(64, 255)), int(np.random.randint(64, 255)), int(np.random.randint(64, 255)))

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.kf.statePost = np.array([initial_position[0], initial_position[1], 0.0, 0.0], dtype=np.float32)
        self.history.append(initial_position)

    def _update_transition(self, dt):
        dt = max(0.01, min(1.0, dt))
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

    def predict(self, dt=1.0):
        self._update_transition(dt)
        prediction = self.kf.predict()
        self.predicted_position = np.array([prediction[0, 0], prediction[1, 0]], dtype=np.float32)
        self.smoothed_position = 0.92 * self.smoothed_position + 0.08 * self.predicted_position
        return self.predicted_position

    def update(self, measurement, timestamp):
        if timestamp > self.last_seen:
            self._update_transition(timestamp - self.last_seen)
        measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]], dtype=np.float32)
        self.kf.correct(measurement)
        self.predicted_position = np.array([measurement[0, 0], measurement[1, 0]], dtype=np.float32)
        self.smoothed_position = 0.8 * self.smoothed_position + 0.2 * self.predicted_position
        self.history.append(tuple(self.smoothed_position))
        self.last_seen = timestamp
        self.age += 1
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1
        self.age += 1
        self.predicted_position = np.array(self.smoothed_position, dtype=np.float32)
        self.history.append(tuple(self.smoothed_position))

    def get_velocity(self):
        state = np.array(self.kf.statePost).ravel()
        if state.shape[0] >= 4:
            vx = float(state[2])
            vy = float(state[3])
        else:
            vx, vy = 0.0, 0.0
        return np.array([vx, vy], dtype=np.float32)


class BallTracker:
    def __init__(self, max_tracks=5, max_missed=12, match_distance=80.0):
        self.max_tracks = max_tracks
        self.max_missed = max_missed
        self.match_distance = match_distance
        self.tracks = []
        self.next_id = 1

    def predict_all(self):
        for track in self.tracks:
            if track.missed_frames == 0:
                track.predict()

    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32))

    def update(self, detections, timestamp):
        self.predict_all()
        if len(self.tracks) == 0:
            for det in detections[: self.max_tracks]:
                self.tracks.append(BallTrack(self.next_id, det, timestamp))
                self.next_id += 1
            return

        cost_matrix = []
        for track in self.tracks:
            row = [self._distance(track.predicted_position, det) for det in detections]
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix, dtype=np.float32)
        assigned_tracks = set()
        assigned_detections = set()

        candidates = []
        for t_idx in range(cost_matrix.shape[0]):
            for d_idx in range(cost_matrix.shape[1]):
                candidates.append((cost_matrix[t_idx, d_idx], t_idx, d_idx))
        candidates.sort(key=lambda x: x[0])

        for cost, t_idx, d_idx in candidates:
            if cost > self.match_distance:
                continue
            if t_idx in assigned_tracks or d_idx in assigned_detections:
                continue
            assigned_tracks.add(t_idx)
            assigned_detections.add(d_idx)
            self.tracks[t_idx].update(detections[d_idx], timestamp)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in assigned_tracks:
                track.mark_missed()

        unmatched_detections = [detections[i] for i in range(len(detections)) if i not in assigned_detections]
        for det in unmatched_detections:
            if len(self.tracks) >= self.max_tracks:
                break
            self.tracks.append(BallTrack(self.next_id, det, timestamp))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.missed_frames <= self.max_missed]

    def get_active_tracks(self):
        return [track for track in self.tracks if track.missed_frames <= self.max_missed]


def color_mask(frame, ball_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if ball_color == "yellow":
        lower = np.array([18, 100, 120], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
    else:
        return None
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    return mask


def detect_round_objects(frame, min_radius=15, max_radius=80, ball_color="none"):
    mask = color_mask(frame, ball_color) if ball_color != "none" else None
    if mask is not None:
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        edges = cv2.Canny(blurred, 50, 150)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        edges = cv2.Canny(blurred, 70, 170)

    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue

        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue
        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_radius or radius > max_radius:
            continue

        if mask is not None:
            mask_area = np.zeros_like(mask)
            cv2.drawContours(mask_area, [cnt], -1, 255, -1)
            color_coverage = cv2.mean(mask, mask=mask_area)[0]
            if color_coverage < 100:
                continue

        circle_area = math.pi * (radius ** 2)
        if area / circle_area < 0.65:
            continue

        detections.append((x, y))

    if len(detections) < 2:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=100,
            param2=35,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            for circle in circles[0]:
                x, y, r = circle
                if r < min_radius or r > max_radius:
                    continue
                detections.append((float(x), float(y)))

    unique = []
    for det in detections:
        if all(np.linalg.norm(np.array(det) - np.array(u)) > 30 for u in unique):
            unique.append(det)

    return unique[:5]


def draw_tracks(frame, tracker):
    for track in tracker.get_active_tracks():
        pos = tuple(map(int, track.smoothed_position))
        cv2.circle(frame, pos, 10, track.color, 2)
        cv2.putText(
            frame,
            f"#{track.id}",
            (pos[0] - 10, pos[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            track.color,
            2,
        )
        pts = list(track.history)
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), track.color, 2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Webcam ball tracker with prediction and occlusion recovery")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for webcam")
    parser.add_argument("--max-balls", type=int, default=5, help="Maximum number of balls to track")
    parser.add_argument("--ball-color", type=str, choices=["none", "yellow"], default="yellow", help="Ball color to filter for")
    parser.add_argument("--flip", dest="flip", action="store_true", help="Flip the display horizontally so the feed is not mirrored")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="Do not flip the display horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--display", action="store_true", help="Display the tracking preview window")
    parser.add_argument("--output", type=str, default=None, help="Optional output video filename")
    return parser.parse_args()


def main():
    args = parse_arguments()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam at index {args.camera}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    tracker = BallTracker(max_tracks=args.max_balls, max_missed=20, match_distance=150.0)
    timestamp = time.time()

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        timestamp = time.time()
        detections = detect_round_objects(frame, ball_color=args.ball_color)
        tracker.update(detections, timestamp)
        draw_tracks(frame, tracker)

        if args.display:
            cv2.imshow("Ball Tracker", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
