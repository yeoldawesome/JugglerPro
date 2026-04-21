#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           J U G G L E   A N A L Y Z E R   v2.0              ║
║      Real-time Computer Vision Juggling Analysis System      ║
╚══════════════════════════════════════════════════════════════╝

Requirements (install with: pip install -r juggling_requirements.txt):
  opencv-python, mediapipe, numpy, scipy, collections, dataclasses
"""

import cv2
import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker, HandLandmarkerOptions
)
from mediapipe.tasks.python.vision.core.image import Image as MPImage, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.optimize import curve_fit, linear_sum_assignment
from collections import deque, defaultdict

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time
import math
import warnings
import os
import urllib.request
warnings.filterwarnings("ignore")

# Auto-download hand_landmarker.task if missing
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Downloaded {MODEL_PATH}.")

# ─────────────────────────────────────────────────────────────
#  CONSTANTS & THEME
# ─────────────────────────────────────────────────────────────

# Neon-on-dark colour palette (BGR)
C_BG         = (10,  10,  18)
C_ACCENT     = (0,  255, 180)   # teal-green
C_WARN       = (0,  180, 255)   # amber
C_DANGER     = (40,  40, 255)   # red
C_PURPLE     = (220,  80, 220)
C_WHITE      = (240, 240, 240)
C_DARK_GRAY  = (60,  60,  70)
C_PANEL      = (20,  22,  35)

BALL_COLORS  = [
    (0,  255, 180),   # teal
    (0,  140, 255),   # orange
    (220, 80, 220),   # purple
    (80, 220, 255),   # yellow
    (255, 100,  80),  # blue
]

TRAIL_LEN        = 40       # frames of trail
MAX_BALLS        = 7
KALMAN_NOISE_COV = 1e-4
LOST_TIMEOUT     = 1.5      # seconds before ball is removed
COLLISION_THRESH = 80       # pixels — warn if balls predicted this close
FPS_SMOOTH       = 20       # frames for FPS averaging

# ─────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class KalmanBallFilter:
    """Per-ball Kalman filter for smooth position + velocity estimation."""
    kf: cv2.KalmanFilter = field(init=False)

    def __post_init__(self):
        kf = cv2.KalmanFilter(6, 2)   # state: x,y,vx,vy,ax,ay
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0, .5*dt**2, 0],
            [0, 1, 0, dt, 0, .5*dt**2],
            [0, 0, 1,  0, dt,  0],
            [0, 0, 0,  1,  0, dt],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov    = np.eye(6, dtype=np.float32) * 1e-3
        kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost       = np.eye(6, dtype=np.float32)
        self.kf = kf

    def predict(self) -> np.ndarray:
        return self.kf.predict()[:2].flatten()

    def correct(self, x: float, y: float) -> np.ndarray:
        m = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kf.correct(m)[:2].flatten()

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])

    @property
    def acceleration(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[4, 0]), float(s[5, 0])


@dataclass
class Ball:
    id: int
    pos: Tuple[float, float]
    color_bgr: Tuple[int, int, int]
    kf: KalmanBallFilter = field(default_factory=KalmanBallFilter)
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    last_seen: float = field(default_factory=time.time)
    height_history: deque = field(default_factory=lambda: deque(maxlen=90))
    peak_height: float = 0.0
    throws: int = 0
    catches: int = 0
    in_flight: bool = False
    lost: bool = False
    missed_frames: int = 0
    predicted_pos: Optional[Tuple[float,float]] = None

    def update(self, x: float, y: float):
        self.kf.correct(x, y)
        pred = self.kf.predict()
        self.pos = (float(pred[0]), float(pred[1]))
        self.trail.append(self.pos)
        self.last_seen = time.time()
        self.lost = False
        self.missed_frames = 0

    def tick_predict(self):
        """Called when no detection — advance Kalman without correction."""
        pred = self.kf.predict()
        self.predicted_pos = (float(pred[0]), float(pred[1]))
        self.missed_frames += 1

    @property
    def velocity(self) -> Tuple[float,float]:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    def future_positions(self, steps=20, dt=1.0) -> List[Tuple[float,float]]:
        """Parabolic trajectory prediction from current state."""
        vx, vy = self.velocity
        ax, ay = self.kf.acceleration
        x, y   = self.pos
        pts = []
        for i in range(1, steps+1):
            t = i * dt
            fx = x + vx*t + 0.5*ax*t**2
            fy = y + vy*t + 0.5*ay*t**2
            pts.append((fx, fy))
        return pts


@dataclass
class HandInfo:
    side: str   # "Left" / "Right"
    landmarks: List[Tuple[float,float]]
    wrist: Tuple[float,float]
    palm_center: Tuple[float,float]
    position: Tuple[float,float] = field(init=False)
    rotation_deg: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float,float] = (0.0, 0.0)
    prev_wrist: Optional[Tuple[float,float]] = None
    hold_count: int = 0    # balls currently held (near palm)

    def __post_init__(self):
        self.position = self.palm_center

    def update_velocity(self, new_wrist: Tuple[float,float], dt: float):
        if self.prev_wrist and dt > 0:
            vx = (new_wrist[0] - self.prev_wrist[0]) / dt
            vy = (new_wrist[1] - self.prev_wrist[1]) / dt
            # EMA smoothing
            a = 0.4
            self.velocity = (
                a*vx + (1-a)*self.velocity[0],
                a*vy + (1-a)*self.velocity[1],
            )
        self.prev_wrist = new_wrist
        self.trail.append(new_wrist)


# ─────────────────────────────────────────────────────────────
#  BALL DETECTOR  (colour + contour + Hough)
# ─────────────────────────────────────────────────────────────

class BallDetector:
    """Detects circular blobs that could be juggling balls."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=35, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray) -> List[Tuple[float,float,float]]:
        """Returns list of (cx, cy, radius) candidates."""
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        candidates = []
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.3,
            minDist=35, param1=70, param2=24,
            minRadius=10, maxRadius=55
        )
        if circles is not None:
            for c in np.round(circles[0]).astype(int):
                candidates.append((float(c[0]), float(c[1]), float(c[2])))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([18, 120, 120]), np.array([40, 255, 255]))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self.kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.kernel)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 7000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if 8 < r < 65:
                candidates.append((cx, cy, r))

        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 7000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if 8 < r < 65:
                candidates.append((cx, cy, r))

        return self._merge(candidates)

    @staticmethod
    def _merge(pts: List[Tuple[float,float,float]],
               threshold: float = 30) -> List[Tuple[float,float,float]]:
        used   = [False] * len(pts)
        merged = []
        for i, (x1, y1, r1) in enumerate(pts):
            if used[i]:
                continue
            group = [(x1, y1, r1)]
            for j, (x2, y2, r2) in enumerate(pts):
                if i == j or used[j]:
                    continue
                if math.hypot(x1-x2, y1-y2) < threshold:
                    group.append((x2, y2, r2))
                    used[j] = True
            mx = np.mean([g[0] for g in group])
            my = np.mean([g[1] for g in group])
            mr = np.mean([g[2] for g in group])
            merged.append((mx, my, mr))
            used[i] = True
        return merged


# ─────────────────────────────────────────────────────────────
#  BALL TRACKER  (Hungarian assignment + persistence)
# ─────────────────────────────────────────────────────────────

class BallTracker:
    def __init__(self):
        self.balls:  Dict[int, Ball] = {}
        self._next_id = 0
        self._assign_threshold = 110   # px — max distance to associate

    def update(self, detections: List[Tuple[float,float,float]],
               frame_h: int,
               hands: Optional[List[HandInfo]] = None) -> Dict[int, Ball]:
        now = time.time()

        for b in self.balls.values():
            b.tick_predict()

        unmatched_det = set(range(len(detections)))
        unmatched_balls = set(self.balls.keys())
        ball_ids = list(self.balls.keys())

        if ball_ids and detections:
            cost = np.zeros((len(ball_ids), len(detections)))
            for bi, bid in enumerate(ball_ids):
                bx, by = self.balls[bid].predicted_pos or self.balls[bid].pos
                for di, (dx, dy, _) in enumerate(detections):
                    cost[bi, di] = math.hypot(bx-dx, by-dy)

            row_ind, col_ind = linear_sum_assignment(cost)
            for bi, di in zip(row_ind, col_ind):
                bid = ball_ids[bi]
                if cost[bi, di] < self._assign_threshold:
                    dx, dy, _ = detections[di]
                    self.balls[bid].update(dx, dy)
                    if di in unmatched_det:
                        unmatched_det.remove(di)
                    if bid in unmatched_balls:
                        unmatched_balls.remove(bid)

        for di in sorted(unmatched_det):
            if len(self.balls) >= MAX_BALLS:
                break
            dx, dy, _ = detections[di]
            color = BALL_COLORS[self._next_id % len(BALL_COLORS)]
            b = Ball(id=self._next_id, pos=(dx, dy), color_bgr=color)
            b.kf.correct(dx, dy)
            self.balls[self._next_id] = b
            self._next_id += 1

        if hands:
            for bid in list(unmatched_balls):
                b = self.balls[bid]
                best_hand = min(hands,
                                key=lambda h: math.hypot(h.palm_center[0]-b.pos[0],
                                                       h.palm_center[1]-b.pos[1]))
                dist = math.hypot(best_hand.palm_center[0]-b.pos[0],
                                  best_hand.palm_center[1]-b.pos[1])
                if dist < 80:
                    b.pos = best_hand.palm_center
                    b.trail.append(b.pos)
                    b.last_seen = now
                    b.lost = False
                    b.missed_frames = 0
                    unmatched_balls.remove(bid)

        to_remove = []
        for bid, b in self.balls.items():
            if bid in unmatched_balls:
                b.lost = b.missed_frames > 2
            else:
                b.lost = False

            age = now - b.last_seen
            if age > LOST_TIMEOUT:
                to_remove.append(bid)
        for bid in to_remove:
            del self.balls[bid]

        for b in self.balls.values():
            _, vy = b.velocity
            norm_y = 1.0 - b.pos[1] / frame_h   # 0=bottom 1=top
            b.height_history.append(norm_y)
            b.peak_height = max(b.peak_height, norm_y)
            # throw / catch detection (sign change in vy)
            if vy < -2:    # moving up
                if not b.in_flight:
                    b.throws += 1
                b.in_flight = True
            elif vy > 2:   # moving down
                b.in_flight = False

        return self.balls


# ─────────────────────────────────────────────────────────────
#  ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────

class JugglingAnalytics:
    def __init__(self):
        self.session_start = time.time()
        self.collision_events: deque = deque(maxlen=50)
        self.arm_pattern_history: deque = deque(maxlen=200)
        self.throw_heights: List[float] = []
        self.total_throws = 0
        self.max_simultaneous = 0
        self.pattern_name = "—"
        self.pattern_confidence = 0.0

    def update(self, balls: Dict[int, Ball],
               hands: List[HandInfo], frame_h: int):
        in_flight = sum(1 for b in balls.values() if b.in_flight and not b.lost)
        self.max_simultaneous = max(self.max_simultaneous, in_flight)

        # ── Collision prediction ──
        self.collision_events.clear()
        ball_list = [b for b in balls.values() if not b.lost]
        for i in range(len(ball_list)):
            for j in range(i+1, len(ball_list)):
                bi, bj = ball_list[i], ball_list[j]
                # check future trajectories
                fi = bi.future_positions(steps=15)
                fj = bj.future_positions(steps=15)
                for step, (pi, pj) in enumerate(zip(fi, fj)):
                    dist = math.hypot(pi[0]-pj[0], pi[1]-pj[1])
                    if dist < COLLISION_THRESH:
                        self.collision_events.append({
                            "ids": (bi.id, bj.id),
                            "step": step,
                            "pos":  ((pi[0]+pj[0])/2, (pi[1]+pj[1])/2),
                            "dist": dist,
                        })
                        break

        # ── Pattern recognition ──
        n = len([b for b in balls.values() if not b.lost])
        if n >= 3:
            in_f = sum(1 for b in balls.values() if b.in_flight and not b.lost)
            if in_f >= n - 1:
                self.pattern_name = f"{n}-ball Cascade" if n % 2 == 1 else f"{n}-ball Fountain"
                self.pattern_confidence = min(0.95, 0.5 + in_f * 0.1)
            elif n == 3 and in_f == 1:
                self.pattern_name = "3-ball Shower"
                self.pattern_confidence = 0.75
            else:
                self.pattern_name = f"{n}-ball pattern"
                self.pattern_confidence = 0.6
        elif n == 2:
            self.pattern_name = "2-ball"
            self.pattern_confidence = 0.8
        elif n == 1:
            self.pattern_name = "1-ball"
            self.pattern_confidence = 0.9
        else:
            self.pattern_name = "—"
            self.pattern_confidence = 0.0

        # ── Arm symmetry ──
        if len(hands) == 2:
            h0, h1 = hands[0], hands[1]
            sym = 1.0 - abs(h0.wrist[1] - h1.wrist[1]) / max(frame_h, 1)
            self.arm_pattern_history.append(sym)

    def reset(self):
        self.session_start = time.time()
        self.collision_events.clear()
        self.arm_pattern_history.clear()
        self.throw_heights.clear()
        self.total_throws = 0
        self.max_simultaneous = 0
        self.pattern_name = "—"
        self.pattern_confidence = 0.0

    @property
    def session_time(self) -> str:
        elapsed = int(time.time() - self.session_start)
        return f"{elapsed//60:02d}:{elapsed%60:02d}"

    @property
    def avg_throw_height_pct(self) -> float:
        if not self.throw_heights:
            return 0.0
        return np.mean(self.throw_heights[-30:]) * 100

    @property
    def arm_symmetry_score(self) -> float:
        if not self.arm_pattern_history:
            return 0.0
        return float(np.mean(list(self.arm_pattern_history))) * 100


# ─────────────────────────────────────────────────────────────
#  RENDERER  (all the pretty visuals)
# ─────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.fps_buf = deque(maxlen=FPS_SMOOTH)
        self._last_t = time.time()
        self.hand_paths = {"Left": deque(), "Right": deque()}
        self.ball_paths = deque()
        # Font sizes for different uses
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    # ── helpers ─────────────────────────────────────────────

    def _text(self, img, text, pos, scale=0.5, color=C_WHITE,
              thickness=1, shadow=True):
        if shadow:
            cv2.putText(img, text,
                (pos[0]+1, pos[1]+1), self.FONT, scale,
                (0,0,0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, text, pos, self.FONT, scale,
                    color, thickness, cv2.LINE_AA)

    def _panel(self, img, x, y, w, h, alpha=0.65):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), C_PANEL, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        cv2.rectangle(img, (x, y), (x+w, y+h), C_DARK_GRAY, 1)

    def _bar(self, img, x, y, w, h, pct, fg=C_ACCENT, bg=C_DARK_GRAY):
        cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
        fill = int(w * max(0, min(1, pct)))
        if fill > 0:
            cv2.rectangle(img, (x, y), (x+fill, y+h), fg, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), C_DARK_GRAY, 1)

    def _dot(self, img, pos, r, color, alpha=1.0):
        overlay = img.copy()
        cv2.circle(overlay, (int(pos[0]), int(pos[1])), r, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    # ── fps ──────────────────────────────────────────────────

    def tick_fps(self) -> float:
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        if dt > 0:
            self.fps_buf.append(1.0 / dt)
        return float(np.mean(self.fps_buf)) if self.fps_buf else 0.0

    # ── draw balls ───────────────────────────────────────────

    def draw_balls(self, img: np.ndarray, balls: Dict[int, Ball]):
        for ball in balls.values():
            c = ball.color_bgr
            pos = ball.predicted_pos if ball.lost else ball.pos
            if pos is None:
                continue
            ix, iy = int(pos[0]), int(pos[1])

            # ── trail ──
            trail = list(ball.trail)
            for k in range(1, len(trail)):
                alpha = k / len(trail)
                tc = tuple(int(ch * alpha * 0.8) for ch in c)
                t0 = (int(trail[k-1][0]), int(trail[k-1][1]))
                t1 = (int(trail[k][0]),   int(trail[k][1]))
                thick = max(1, int(3 * alpha))
                cv2.line(img, t0, t1, tc, thick, cv2.LINE_AA)

            # ── predicted trajectory (parabola) ──
            if not ball.lost:
                future = ball.future_positions(steps=18, dt=1.5)
                for k in range(1, len(future)):
                    alpha = 1.0 - k / len(future)
                    pc = tuple(int(ch * alpha * 0.5) for ch in c)
                    p0 = (int(future[k-1][0]), int(future[k-1][1]))
                    p1 = (int(future[k][0]),   int(future[k][1]))
                    if (0 <= p0[0] < self.w and 0 <= p0[1] < self.h and
                        0 <= p1[0] < self.w and 0 <= p1[1] < self.h):
                        cv2.line(img, p0, p1, pc, 1, cv2.LINE_AA)

            # ── glow ring ──
            glow_r = 22 if not ball.lost else 16
            for gr in range(glow_r, glow_r-8, -2):
                a = (glow_r - gr) / 8 * 0.15
                ov = img.copy()
                cv2.circle(ov, (ix, iy), gr, c, -1)
                cv2.addWeighted(ov, a, img, 1-a, 0, img)

            # ── main circle ──
            style = C_DARK_GRAY if ball.lost else c
            cv2.circle(img, (ix, iy), 16, style, -1, cv2.LINE_AA)
            cv2.circle(img, (ix, iy), 16, C_WHITE, 1, cv2.LINE_AA)

            # ── ID label ──
            self._text(img, str(ball.id),
                       (ix-5, iy+5), 0.5, C_BG, 2, shadow=False)

            # ── velocity arrow ──
            vx, vy = ball.velocity
            spd = ball.speed
            if spd > 1.5 and not ball.lost:
                scale = min(30, spd * 3) / max(spd, 1)
                ex = int(ix + vx * scale)
                ey = int(iy + vy * scale)
                cv2.arrowedLine(img, (ix, iy), (ex, ey), c, 2,
                                cv2.LINE_AA, tipLength=0.4)

    # ── draw hands ───────────────────────────────────────────

    def draw_hands(self, img: np.ndarray, hands: List[HandInfo]):
        for hand in hands:
            c  = C_ACCENT if hand.side == "Right" else C_PURPLE
            px, py = int(hand.position[0]), int(hand.position[1])
            wx, wy = int(hand.wrist[0]), int(hand.wrist[1])

            # general hand position
            cv2.circle(img, (px, py), 18, c, 2, cv2.LINE_AA)
            cv2.circle(img, (wx, wy), 10, c, -1, cv2.LINE_AA)
            self._text(img, hand.side[0], (px-8, py-24), 0.48, c, 1)

            # rotation arrow
            angle_rad = math.radians(hand.rotation_deg)
            length = 50
            ex = int(px + math.cos(angle_rad) * length)
            ey = int(py + math.sin(angle_rad) * length)
            cv2.arrowedLine(img, (px, py), (ex, ey), c, 2,
                            cv2.LINE_AA, tipLength=0.3)

            # quick direction marker
            self._text(img, f"{hand.rotation_deg:+.0f}°",
                       (px-12, py+28), 0.35, C_WHITE, 1)

    # ── collision warnings ───────────────────────────────────

    def draw_collisions(self, img: np.ndarray,
                        events: deque):
        for ev in events:
            cx, cy = int(ev["pos"][0]), int(ev["pos"][1])
            step   = ev["step"]
            urgency = 1.0 - step / 15.0   # 1=imminent 0=far
            r      = int(30 + 20 * urgency)
            col    = (
                int(C_WARN[0] * (1-urgency) + C_DANGER[0] * urgency),
                int(C_WARN[1] * (1-urgency) + C_DANGER[1] * urgency),
                int(C_WARN[2] * (1-urgency) + C_DANGER[2] * urgency),
            )
            # pulsing ring
            pulse = int(r + 5 * math.sin(time.time() * 10))
            ov = img.copy()
            cv2.circle(ov, (cx, cy), pulse, col, 2)
            cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
            if urgency > 0.7:
                self._text(img, "⚠ COLLISION",
                           (cx-40, cy-pulse-6), 0.4, col, 1)

    # ── left stats panel ─────────────────────────────────────

    def draw_left_panel(self, img: np.ndarray,
                        balls: Dict[int, Ball],
                        analytics: JugglingAnalytics,
                        fps: float):
        PW, PH = 200, self.h
        self._panel(img, 0, 0, PW, PH, alpha=0.75)

        # Header
        cv2.line(img, (0, 0), (PW, 0), C_ACCENT, 2)
        self._text(img, "JUGGLE ANALYZER", (8, 24), 0.52, C_ACCENT, 1)
        self._text(img, "v2.0", (148, 24), 0.35, C_DARK_GRAY, 1)
        cv2.line(img, (8, 32), (PW-8, 32), C_DARK_GRAY, 1)

        y = 52

        def row(label, val, color=C_WHITE):
            nonlocal y
            self._text(img, label, (10, y), 0.38, C_DARK_GRAY, 1)
            self._text(img, str(val), (10, y+14), 0.48, color, 1)
            y += 36

        row("SESSION", analytics.session_time, C_ACCENT)
        row("PATTERN", analytics.pattern_name, C_WHITE)
        row("BALLS", f"{len([b for b in balls.values() if not b.lost])}", C_ACCENT)
        row("MAX SIMULTANEOUS", str(analytics.max_simultaneous))
        row("FPS", f"{fps:.1f}", C_ACCENT if fps > 25 else C_WARN)

        # Throw height bar
        self._text(img, "THROW HEIGHT", (10, y), 0.38, C_DARK_GRAY, 1)
        y += 14
        avg_h = analytics.avg_throw_height_pct / 100.0
        self._bar(img, 10, y, PW-20, 10, avg_h)
        self._text(img, f"{analytics.avg_throw_height_pct:.0f}%",
                   (10, y+22), 0.38, C_WHITE, 1)
        y += 34

        # Arm symmetry
        sym = analytics.arm_symmetry_score / 100.0
        self._text(img, "ARM SYMMETRY", (10, y), 0.38, C_DARK_GRAY, 1)
        y += 14
        col = C_ACCENT if sym > 0.7 else C_WARN
        self._bar(img, 10, y, PW-20, 10, sym, fg=col)
        self._text(img, f"{analytics.arm_symmetry_score:.0f}%",
                   (10, y+22), 0.38, C_WHITE, 1)
        y += 36

        # Pattern confidence
        conf = analytics.pattern_confidence
        self._text(img, "PATTERN CONF.", (10, y), 0.38, C_DARK_GRAY, 1)
        y += 14
        self._bar(img, 10, y, PW-20, 10, conf,
                  fg=C_ACCENT if conf > 0.7 else C_WARN)
        self._text(img, f"{conf*100:.0f}%",
                   (10, y+22), 0.38, C_WHITE, 1)
        y += 36

        # Per-ball breakdown
        cv2.line(img, (8, y), (PW-8, y), C_DARK_GRAY, 1); y += 10
        self._text(img, "PER BALL", (10, y), 0.4, C_ACCENT, 1); y += 18
        for ball in balls.values():
            if ball.lost:
                continue
            vx, vy = ball.velocity
            c_dot  = ball.color_bgr
            cv2.circle(img, (18, y-4), 5, c_dot, -1)
            self._text(img, f"#{ball.id}  spd:{ball.speed:.0f}",
                       (28, y), 0.36, C_WHITE, 1)
            y += 14
            self._text(img,
                       f"   thr:{ball.throws}  "
                       f"h:{ball.peak_height*100:.0f}%",
                       (10, y), 0.33, C_DARK_GRAY, 1)
            y += 16
            if y > self.h - 30:
                break

        # Bottom hint
        self._text(img, "Press Q to quit",
                   (10, self.h-12), 0.35, C_DARK_GRAY, 1)

    # ── right overlay panel ───────────────────────────────────

    def draw_right_panel(self, img: np.ndarray,
                         balls: Dict[int, Ball],
                         hands: List[HandInfo],
                         analytics: JugglingAnalytics):
        PW = 180
        x0 = self.w - PW
        self._panel(img, x0, 0, PW, 300, alpha=0.72)
        cv2.line(img, (x0, 0), (self.w, 0), C_PURPLE, 2)

        y = 20
        self._text(img, "HAND ANALYSIS", (x0+8, y), 0.45, C_PURPLE, 1); y += 20
        cv2.line(img, (x0+8, y), (self.w-8, y), C_DARK_GRAY, 1); y += 12

        for hand in hands:
            vx, vy = hand.velocity
            c = C_ACCENT if hand.side == "Right" else C_PURPLE
            self._text(img, f"{hand.side} Hand", (x0+8, y), 0.42, c, 1); y += 16
            spd = math.hypot(vx, vy)
            self._text(img, f"Speed: {spd:.0f}px/f",
                       (x0+10, y), 0.35, C_WHITE, 1); y += 13
            self._text(img, f"Rotation: {hand.rotation_deg:+.0f}°",
                       (x0+10, y), 0.35, C_WHITE, 1); y += 13
            self._text(img, f"Holds: {hand.hold_count} ball(s)",
                       (x0+10, y), 0.35, C_DARK_GRAY, 1); y += 18

        if analytics.collision_events:
            self._text(img, "⚠ COLLISION RISK",
                       (x0+8, y), 0.42, C_DANGER, 1); y += 16
            for ev in list(analytics.collision_events)[:3]:
                ids = ev["ids"]
                self._text(img, f"  #{ids[0]}↔#{ids[1]} in {ev['step']}f",
                           (x0+8, y), 0.35, C_WARN, 1); y += 13

    # ── height mini-graph ─────────────────────────────────────

    def draw_height_graph(self, img: np.ndarray,
                          balls: Dict[int, Ball]):
        GW, GH = 180, 70
        GX = self.w - GW - 5
        GY = self.h - GH - 30
        self._panel(img, GX-5, GY-15, GW+10, GH+25, alpha=0.7)
        self._text(img, "HEIGHT  (last 90 frames)",
                   (GX, GY-4), 0.33, C_DARK_GRAY, 1)
        cv2.rectangle(img, (GX, GY), (GX+GW, GY+GH), C_DARK_GRAY, 1)

        for ball in balls.values():
            if ball.lost or len(ball.height_history) < 2:
                continue
            hist = list(ball.height_history)
            pts  = []
            for i, h in enumerate(hist):
                px = GX + int(i * GW / max(len(hist)-1, 1))
                py = GY + GH - int(h * GH)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], ball.color_bgr, 1, cv2.LINE_AA)

    # ── top status bar ───────────────────────────────────────

    def draw_topbar(self, img, analytics, fps):
        cv2.rectangle(img, (200, 0), (self.w-180, 28), C_PANEL, -1)
        cv2.line(img, (200, 28), (self.w-180, 28), C_DARK_GRAY, 1)
        status = (f"  {analytics.pattern_name}   │   "
                  f"Time {analytics.session_time}   │   "
                  f"FPS {fps:.1f}   │   "
                  f"Max balls: {analytics.max_simultaneous}")
        self._text(img, status, (210, 18), 0.45, C_WHITE, 1)

    def clear_paths(self):
        self.hand_paths = {"Left": deque(), "Right": deque()}
        self.ball_paths.clear()

    def record_paths(self, balls: Dict[int, Ball], hands: List[HandInfo], now: float):
        # keep only the last 30 seconds of motion history
        max_age = 30.0
        for hand in hands:
            self.hand_paths[hand.side].append((now, hand.position))
        for side in list(self.hand_paths.keys()):
            self.hand_paths[side] = deque(
                [(t, pos) for t, pos in self.hand_paths[side]
                 if now - t <= max_age], maxlen=2000)

        for ball in balls.values():
            if ball.lost:
                continue
            self.ball_paths.append((now, ball.id, ball.pos, ball.color_bgr))
        self.ball_paths = deque(
            [(t, bid, pos, c) for t, bid, pos, c in self.ball_paths
             if now - t <= max_age], maxlen=2000)

    def draw_path_history_panel(self, img: np.ndarray):
        PW, PH = 210, 210
        GX = self.w - PW - 190
        GY = 10
        self._panel(img, GX, GY, PW, PH, alpha=0.78)
        self._text(img, "HAND + BALL PATHS", (GX+10, GY+18), 0.36, C_WHITE, 1)
        self._text(img, "last 30 sec", (GX+10, GY+34), 0.28, C_DARK_GRAY, 1)

        inner_x = GX + 10
        inner_y = GY + 44
        inner_w = PW - 20
        inner_h = PH - 54
        cv2.rectangle(img, (inner_x, inner_y),
                      (inner_x+inner_w, inner_y+inner_h), C_DARK_GRAY, 1)

        def project(pt):
            return (
                inner_x + int(pt[0] / max(self.w, 1) * inner_w),
                inner_y + int(pt[1] / max(self.h, 1) * inner_h)
            )

        # draw ball paths grouped by ball id
        ball_segments = {}
        for _, bid, pos, color in self.ball_paths:
            p = project(pos)
            if bid in ball_segments:
                prev_p, prev_c = ball_segments[bid]
                cv2.line(img, prev_p, p, color, 1, cv2.LINE_AA)
            ball_segments[bid] = (p, color)
            cv2.circle(img, p, 2, color, -1, cv2.LINE_AA)

        # draw hand paths
        for side, path in self.hand_paths.items():
            color = C_ACCENT if side == "Right" else C_PURPLE
            points = [project(pos) for _, pos in path]
            for i in range(1, len(points)):
                alpha = i / max(len(points)-1, 1)
                line_color = tuple(int(ch * (0.3 + 0.7*alpha)) for ch in color)
                cv2.line(img, points[i-1], points[i], line_color, 2, cv2.LINE_AA)
            if points:
                cv2.circle(img, points[-1], 4, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

class JuggleAnalyzerApp:
    def __init__(self):
        # MediaPipe Tasks HandLandmarker
        model_path = "hand_landmarker.task"
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands_model = HandLandmarker.create_from_options(options)

        self.detector  = BallDetector()
        self.tracker   = BallTracker()
        self.analytics = JugglingAnalytics()

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot open webcam!")

        h, w = frame.shape[:2]
        self.renderer = Renderer(w, h)
        self._prev_time = time.time()
        self._prev_hand_wrists: Dict[str, Tuple[float,float]] = {}

    def _parse_hands(self, results, frame_w: int, frame_h: int) -> List[HandInfo]:
        hands_out = []
        if not results.hand_landmarks:
            return hands_out

        for lm_list, handedness in zip(results.hand_landmarks, results.handedness):
            side = handedness[0].category_name  # "Left"/"Right"
            lms  = [(lm.x * frame_w, lm.y * frame_h) for lm in lm_list]
            wrist = lms[0]
            palm  = (
                np.mean([lms[i][0] for i in [0,5,9,13,17]]),
                np.mean([lms[i][1] for i in [0,5,9,13,17]]),
            )
            # approximate hand orientation using the direction from the wrist to average finger base positions
            avg_fingers = np.mean([lms[i] for i in [5, 9, 13, 17]], axis=0)
            dx, dy = avg_fingers[0] - wrist[0], avg_fingers[1] - wrist[1]
            rotation_deg = math.degrees(math.atan2(dy, dx))
            hi = HandInfo(
                side=side,
                landmarks=lms,
                wrist=wrist,
                palm_center=palm,
                rotation_deg=rotation_deg,
            )
            now = time.time()
            dt  = now - self._prev_time if now != self._prev_time else 1/30
            hi.update_velocity(wrist, dt)
            hands_out.append(hi)
        return hands_out

    def _count_balls_near_hands(self,
                                 balls: Dict[int, Ball],
                                 hands: List[HandInfo]):
        for hand in hands:
            count = 0
            px, py = hand.palm_center
            for ball in balls.values():
                if ball.lost:
                    continue
                bx, by = ball.pos
                if math.hypot(bx-px, by-py) < 60:
                    count += 1
            hand.hold_count = count

    def run(self):
        print("\n╔══════════════════════════════════╗")
        print("║  Juggle Analyzer  — running!     ║")
        print("║  Press Q to quit                 ║")
        print("╚══════════════════════════════════╝\n")

        cv2.namedWindow("Juggle Analyzer", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            h, w  = frame.shape[:2]

            # ── Darken background slightly for overlay clarity ──
            overlay_bg = frame.copy()
            cv2.addWeighted(overlay_bg, 0.85,
                            np.zeros_like(frame), 0.15, 0, frame)

            # ── Hand detection ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(ImageFormat.SRGB, rgb)
            hand_results = self.hands_model.detect(mp_image)
            hands = self._parse_hands(hand_results, w, h)

            # ── Ball detection + tracking ──
            detections = self.detector.detect(frame)
            balls      = self.tracker.update(detections, h, hands)

            # ── Analytics ──
            self._count_balls_near_hands(balls, hands)
            self.analytics.update(balls, hands, h)
            self.renderer.record_paths(balls, hands, time.time())

            # ── Collect throw heights ──
            for b in balls.values():
                if b.in_flight and not b.lost:
                    norm = 1.0 - b.pos[1] / h
                    self.analytics.throw_heights.append(norm)

            # ── Render ──
            fps = self.renderer.tick_fps()
            self.renderer.draw_balls(frame, balls)
            self.renderer.draw_hands(frame, hands)
            self.renderer.draw_collisions(
                frame, self.analytics.collision_events)
            self.renderer.draw_left_panel(
                frame, balls, self.analytics, fps)
            self.renderer.draw_right_panel(
                frame, balls, hands, self.analytics)
            self.renderer.draw_path_history_panel(frame)
            self.renderer.draw_height_graph(frame, balls)
            self.renderer.draw_topbar(frame, self.analytics, fps)

            self._prev_time = time.time()

            cv2.imshow("Juggle Analyzer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('c'), ord('C')):
                self.analytics.reset()
                self.renderer.clear_paths()
                for b in balls.values():
                    b.height_history.clear()
                    b.throws = 0
                    b.peak_height = 0.0
                print("📊 Charts cleared")

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        # No close() for HandLandmarker in Tasks API
        cv2.destroyAllWindows()
        print("\n📊 Session Summary")
        print(f"   Session time   : {self.analytics.session_time}")
        print(f"   Pattern        : {self.analytics.pattern_name}")
        print(f"   Max balls air  : {self.analytics.max_simultaneous}")
        print(f"   Avg throw ht   : {self.analytics.avg_throw_height_pct:.1f}%")
        print(f"   Arm symmetry   : {self.analytics.arm_symmetry_score:.1f}%")
        print()


if __name__ == "__main__":
    app = JuggleAnalyzerApp()
    app.run()
