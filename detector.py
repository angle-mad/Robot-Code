import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "auto").lower()
VALID_CAMERA_BACKENDS = ("auto", "picamera2", "opencv")
PICAMERA2_SWAP_RED_BLUE = os.environ.get("PICAMERA2_SWAP_RED_BLUE", "true").lower()
ENABLE_ACTUATORS_REQUEST = os.environ.get("ENABLE_ACTUATORS", "false").lower()
VALID_BOOLEAN_OPTIONS = ("true", "false")
HEADLESS_REQUEST = os.environ.get("HEADLESS", "auto").lower()
VALID_HEADLESS_OPTIONS = ("auto", "true", "false")
STEERING_CHANNEL = int(os.environ.get("STEERING_CHANNEL", "0"))
THROTTLE_CHANNEL = int(os.environ.get("THROTTLE_CHANNEL", "1"))
STEERING_CENTER_DEGREES = float(os.environ.get("STEERING_CENTER_DEGREES", "110"))
STEERING_LEFT_DEGREES = float(os.environ.get("STEERING_LEFT_DEGREES", "50"))
STEERING_RIGHT_DEGREES = float(os.environ.get("STEERING_RIGHT_DEGREES", "170"))
STEERING_SERVO_MIN_DEGREES = float(os.environ.get("STEERING_SERVO_MIN_DEGREES", "0"))
STEERING_SERVO_MAX_DEGREES = float(os.environ.get("STEERING_SERVO_MAX_DEGREES", "180"))
STEERING_SERVO_MIN_US = int(os.environ.get("STEERING_SERVO_MIN_US", "500"))
STEERING_SERVO_MAX_US = int(os.environ.get("STEERING_SERVO_MAX_US", "2500"))
THROTTLE_NEUTRAL_US = int(os.environ.get("THROTTLE_NEUTRAL_US", "1500"))
THROTTLE_FORWARD_US = int(os.environ.get("THROTTLE_FORWARD_US", "1600"))
THROTTLE_REVERSE_US = int(os.environ.get("THROTTLE_REVERSE_US", "1400"))
ESC_ARM_SECONDS = float(os.environ.get("ESC_ARM_SECONDS", "3.0"))
THROTTLE_HARD_LIMIT = float(os.environ.get("THROTTLE_HARD_LIMIT", "0.12"))
THROTTLE_MIN_ACTIVE = float(os.environ.get("THROTTLE_MIN_ACTIVE", "0.06"))
THROTTLE_ALLOW_REVERSE = os.environ.get("THROTTLE_ALLOW_REVERSE", "false").lower()
ENABLE_THROTTLE_REQUEST = os.environ.get("ENABLE_THROTTLE", "false").lower()
MAX_TRIAL_THROTTLE = float(os.environ.get("MAX_TRIAL_THROTTLE", "0.10"))
STEERING_GAIN = float(os.environ.get("STEERING_GAIN", "1.25"))
STEERING_DEADBAND = float(os.environ.get("STEERING_DEADBAND", "0.06"))
CLOSE_BALL_AREA_RATIO = float(os.environ.get("CLOSE_BALL_AREA_RATIO", "0.18"))
LOST_TARGET_TIMEOUT = float(os.environ.get("LOST_TARGET_TIMEOUT", "0.5"))
TELEMETRY_INTERVAL = float(os.environ.get("TELEMETRY_INTERVAL", "0.25"))
TUI_REQUEST = os.environ.get("TUI", "true").lower()
TUI_INTERVAL = float(os.environ.get("TUI_INTERVAL", "0.1"))
MIN_BALL_AREA_RATIO = float(os.environ.get("MIN_BALL_AREA_RATIO", "0.004"))
MIN_BALL_AREA_TOP_SCALE = float(os.environ.get("MIN_BALL_AREA_TOP_SCALE", "0.25"))
MAX_BALL_AREA_RATIO = float(os.environ.get("MAX_BALL_AREA_RATIO", "0.15"))
MIN_BALL_CIRCULARITY = float(os.environ.get("MIN_BALL_CIRCULARITY", "0.48"))
MIN_BALL_CIRCLE_FILL = float(os.environ.get("MIN_BALL_CIRCLE_FILL", "0.50"))
TRIANGLE_APPROX_EPSILON = float(os.environ.get("TRIANGLE_APPROX_EPSILON", "0.04"))
AUTO_CALIBRATE_REQUEST = os.environ.get("AUTO_CALIBRATE", "true").lower()
AUTO_CALIBRATION_INTERVAL = float(os.environ.get("AUTO_CALIBRATION_INTERVAL", "1.0"))
AUTO_CALIBRATION_MAX_COLORS = int(os.environ.get("AUTO_CALIBRATION_MAX_COLORS", "8"))
AUTO_CALIBRATION_MIN_AREA_RATIO = float(os.environ.get("AUTO_CALIBRATION_MIN_AREA_RATIO", "0.006"))
AUTO_CALIBRATION_SATURATION_MIN = int(os.environ.get("AUTO_CALIBRATION_SATURATION_MIN", "35"))
AUTO_CALIBRATION_VALUE_MIN = int(os.environ.get("AUTO_CALIBRATION_VALUE_MIN", "70"))
AUTO_CALIBRATION_HUE_PADDING = int(os.environ.get("AUTO_CALIBRATION_HUE_PADDING", "8"))
AUTO_CALIBRATION_SATURATION_PADDING = int(os.environ.get("AUTO_CALIBRATION_SATURATION_PADDING", "45"))
AUTO_CALIBRATION_VALUE_PADDING = int(os.environ.get("AUTO_CALIBRATION_VALUE_PADDING", "45"))
AUTO_CALIBRATION_MERGE_HUE_DISTANCE = int(os.environ.get("AUTO_CALIBRATION_MERGE_HUE_DISTANCE", "10"))
KNOWN_COLOR_HUE_PADDING = int(os.environ.get("KNOWN_COLOR_HUE_PADDING", "12"))
KNOWN_COLOR_SATURATION_MIN = int(os.environ.get("KNOWN_COLOR_SATURATION_MIN", "45"))
KNOWN_COLOR_VALUE_MIN = int(os.environ.get("KNOWN_COLOR_VALUE_MIN", "60"))
TARGET_DISTANCE_WEIGHT = float(os.environ.get("TARGET_DISTANCE_WEIGHT", "0.75"))
TARGET_CLUSTER_WEIGHT = float(os.environ.get("TARGET_CLUSTER_WEIGHT", "0.25"))
TARGET_AREA_WEIGHT = float(os.environ.get("TARGET_AREA_WEIGHT", "0.08"))
TARGET_CENTER_WEIGHT = float(os.environ.get("TARGET_CENTER_WEIGHT", "0.03"))
TARGET_CLUSTER_RADIUS_RATIO = float(os.environ.get("TARGET_CLUSTER_RADIUS_RATIO", "0.22"))
TARGET_LOCK_RADIUS_RATIO = float(os.environ.get("TARGET_LOCK_RADIUS_RATIO", "0.18"))
TARGET_SWITCH_MARGIN = float(os.environ.get("TARGET_SWITCH_MARGIN", "0.18"))
TARGET_SWITCH_FRAMES = int(os.environ.get("TARGET_SWITCH_FRAMES", "3"))
TARGET_HOLD_SECONDS = float(os.environ.get("TARGET_HOLD_SECONDS", "0.35"))
TARGET_SMOOTHING = float(os.environ.get("TARGET_SMOOTHING", "0.35"))
STEERING_SLEW_RATE = float(os.environ.get("STEERING_SLEW_RATE", "2.0"))


def parse_bool(name, value):
    if value not in VALID_BOOLEAN_OPTIONS:
        raise ValueError(
            name
            + " must be one of: "
            + ", ".join(VALID_BOOLEAN_OPTIONS)
        )

    return value == "true"


def running_without_display():
    if HEADLESS_REQUEST not in VALID_HEADLESS_OPTIONS:
        raise ValueError(
            "HEADLESS must be one of: "
            + ", ".join(VALID_HEADLESS_OPTIONS)
        )

    if HEADLESS_REQUEST == "true":
        return True

    if HEADLESS_REQUEST == "false":
        return False

    return os.name != "nt" and os.environ.get("DISPLAY", "") == ""


HEADLESS = running_without_display()
SWAP_PICAMERA2_RED_BLUE = parse_bool(
    "PICAMERA2_SWAP_RED_BLUE",
    PICAMERA2_SWAP_RED_BLUE
)
ENABLE_ACTUATORS = parse_bool(
    "ENABLE_ACTUATORS",
    ENABLE_ACTUATORS_REQUEST
)
AUTO_CALIBRATE = parse_bool(
    "AUTO_CALIBRATE",
    AUTO_CALIBRATE_REQUEST
)
TUI_ENABLED = parse_bool("TUI", TUI_REQUEST)
ALLOW_REVERSE_THROTTLE = parse_bool("THROTTLE_ALLOW_REVERSE", THROTTLE_ALLOW_REVERSE)
ENABLE_THROTTLE = parse_bool("ENABLE_THROTTLE", ENABLE_THROTTLE_REQUEST)


@dataclass
class VisionTarget:
    label: str
    center_x: int
    center_y: int
    area: float
    confidence: float
    box: tuple
    radius: int
    color: tuple


@dataclass
class DriveCommand:
    steering: float
    throttle: float
    mode: str
    reason: str


@dataclass
class AutoCalibrationProfile:
    name: str
    hue: int
    color_range: dict
    box_color: tuple
    last_seen_time: float
    observations: int


@dataclass
class DetectionDebug:
    active_colors: int = 0
    masks_checked: int = 0
    contours_seen: int = 0
    rejected_small: int = 0
    rejected_large: int = 0
    rejected_shape: int = 0
    rejected_samples: int = 0
    rejected_overlap: int = 0
    accepted: int = 0
    auto_candidates: int = 0
    auto_profiles: int = 0
    priority_score: float = 0.0
    priority_distance: float = 0.0
    priority_cluster: float = 0.0
    priority_area: float = 0.0
    priority_center: float = 0.0
    priority_neighbors: int = 0
    raw_target_count: int = 0
    stable_target_locked: bool = False
    stable_target_held: bool = False
    stable_target_label: str = "none"
    stable_target_age: float = 0.0
    switch_candidate_frames: int = 0
    raw_steering: float = 0.0
    smoothed_steering: float = 0.0
    steering_limited: bool = False


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalized_to_pulse(command, minimum_us, center_us, maximum_us):
    command = clamp(command, -1.0, 1.0)

    if command < 0:
        return int(center_us + (center_us - minimum_us) * command)

    return int(center_us + (maximum_us - center_us) * command)


def steering_degrees_to_pulse_us(degrees):
    degrees = clamp(
        degrees,
        min(STEERING_SERVO_MIN_DEGREES, STEERING_SERVO_MAX_DEGREES),
        max(STEERING_SERVO_MIN_DEGREES, STEERING_SERVO_MAX_DEGREES)
    )
    degree_span = STEERING_SERVO_MAX_DEGREES - STEERING_SERVO_MIN_DEGREES

    if degree_span == 0:
        raise ValueError("STEERING_SERVO_MIN_DEGREES and STEERING_SERVO_MAX_DEGREES must differ")

    servo_position = (degrees - STEERING_SERVO_MIN_DEGREES) / degree_span
    pulse_span = STEERING_SERVO_MAX_US - STEERING_SERVO_MIN_US

    return int(round(STEERING_SERVO_MIN_US + servo_position * pulse_span))


def normalized_to_steering_pulse(command):
    command = clamp(command, -1.0, 1.0)

    if command < 0:
        steering_degrees = STEERING_CENTER_DEGREES + (
            STEERING_CENTER_DEGREES - STEERING_LEFT_DEGREES
        ) * command
    else:
        steering_degrees = STEERING_CENTER_DEGREES + (
            STEERING_RIGHT_DEGREES - STEERING_CENTER_DEGREES
        ) * command

    return steering_degrees_to_pulse_us(steering_degrees)


def normalize_throttle_for_esc(command):
    if not ENABLE_THROTTLE:
        return 0.0

    command = clamp(command, -1.0, 1.0)

    if command > 0:
        throttle = min(command, THROTTLE_HARD_LIMIT)

        if throttle > 0 and throttle < THROTTLE_MIN_ACTIVE:
            throttle = min(THROTTLE_MIN_ACTIVE, THROTTLE_HARD_LIMIT)

        return throttle

    if command < 0 and ALLOW_REVERSE_THROTTLE:
        return max(command, -THROTTLE_HARD_LIMIT)

    return 0.0


def normalized_to_esc_throttle_pulse(command):
    return normalized_to_pulse(
        normalize_throttle_for_esc(command),
        THROTTLE_REVERSE_US,
        THROTTLE_NEUTRAL_US,
        THROTTLE_FORWARD_US
    )


class TuiDashboard:
    def __init__(self):
        self.enabled = TUI_ENABLED and sys.stdout.isatty()
        self.screen = None
        self.last_draw_time = 0

        if not self.enabled:
            if TUI_ENABLED:
                print("TUI disabled because stdout is not a terminal.")
            return

        try:
            import curses
        except ImportError:
            print("TUI disabled because curses is unavailable.")
            self.enabled = False
            return

        self.curses = curses
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self.screen.nodelay(True)
        self.screen.keypad(False)

    def close(self):
        if not self.enabled:
            return

        self.curses.nocbreak()
        self.curses.echo()
        try:
            self.curses.curs_set(1)
        except self.curses.error:
            pass
        self.curses.endwin()
        self.enabled = False

    def draw(self, frame, best_target, command, debug, auto_colors, current_time, fps):
        if not self.enabled:
            return

        if current_time - self.last_draw_time < TUI_INTERVAL:
            return

        self.last_draw_time = current_time
        height, width = self.screen.getmaxyx()
        lines = self.make_lines(frame, best_target, command, debug, auto_colors, fps, width)
        self.screen.erase()

        for index, line in enumerate(lines[:height - 1]):
            self.screen.addstr(index, 0, line[:max(0, width - 1)])

        self.screen.refresh()

    def make_lines(self, frame, best_target, command, debug, auto_colors, fps, terminal_width):
        lines = [
            "Robot Code TUI",
            "==============",
            "",
            "[Status]",
            "camera=" + CAMERA_BACKEND
            + " frame=" + str(frame.shape[1]) + "x" + str(frame.shape[0])
            + " fps=" + str(round(fps, 1))
            + " headless=" + str(HEADLESS)
            + " actuators=" + str(ENABLE_ACTUATORS),
            "auto_calibrate=" + str(AUTO_CALIBRATE)
            + " tui_interval=" + str(TUI_INTERVAL)
            + " ctrl-c exits and neutralizes outputs",
            "",
            "[Target]",
        ]

        if best_target is None:
            lines.extend([
                "selected=none",
                "position=n/a area=n/a confidence=n/a",
                "stable label=" + debug.stable_target_label
                + " locked=" + str(debug.stable_target_locked)
                + " held=" + str(debug.stable_target_held)
                + " age=" + str(round(debug.stable_target_age, 2)),
                "switch_frames=" + str(debug.switch_candidate_frames)
                + " raw_targets=" + str(debug.raw_target_count),
            ])
        else:
            lines.extend([
                "selected=" + best_target.label,
                "position=x:" + str(best_target.center_x)
                + " y:" + str(best_target.center_y)
                + " radius:" + str(best_target.radius),
                "area=" + str(int(best_target.area))
                + " confidence=" + str(round(best_target.confidence, 2)),
                "priority score=" + str(round(debug.priority_score, 3))
                + " distance=" + str(round(debug.priority_distance, 3))
                + " cluster=" + str(round(debug.priority_cluster, 3)),
                "priority area=" + str(round(debug.priority_area, 3))
                + " center=" + str(round(debug.priority_center, 3))
                + " neighbors=" + str(debug.priority_neighbors),
                "stable label=" + debug.stable_target_label
                + " locked=" + str(debug.stable_target_locked)
                + " held=" + str(debug.stable_target_held)
                + " age=" + str(round(debug.stable_target_age, 2)),
                "switch_frames=" + str(debug.switch_candidate_frames)
                + " raw_targets=" + str(debug.raw_target_count),
            ])

        lines.extend([
            "",
            "[Drive]",
            "mode=" + command.mode
            + " steering=" + str(round(command.steering, 3))
            + " throttle=" + str(round(command.throttle, 3)),
            "reason=" + command.reason,
            "raw_steering=" + str(round(debug.raw_steering, 3))
            + " smoothed_steering=" + str(round(debug.smoothed_steering, 3))
            + " slew_limited=" + str(debug.steering_limited),
            self.make_steering_bar(command.steering, terminal_width),
            "esc safe_throttle=" + str(round(normalize_throttle_for_esc(command.throttle), 3))
            + " throttle_us=" + str(getattr(actuators, "last_throttle_us", None))
            + " armed=" + str(getattr(actuators, "esc_armed", False))
            + " enabled=" + str(ENABLE_THROTTLE),
            "",
            "[Vision]",
            "active_colors=" + str(debug.active_colors)
            + " masks=" + str(debug.masks_checked)
            + " contours=" + str(debug.contours_seen)
            + " accepted=" + str(debug.accepted),
            "reject_small=" + str(debug.rejected_small)
            + " reject_large=" + str(debug.rejected_large)
            + " reject_shape=" + str(debug.rejected_shape)
            + " reject_samples=" + str(debug.rejected_samples)
            + " reject_overlap=" + str(debug.rejected_overlap),
            "",
            "[Auto Calibration]",
            "candidates=" + str(debug.auto_candidates)
            + " profiles=" + str(debug.auto_profiles),
        ])

        if len(auto_colors) == 0:
            lines.append("profiles=none")
        else:
            profile_names = [
                color["name"]
                for color in auto_colors[:8]
            ]
            lines.append("profiles=" + ", ".join(profile_names))

        lines.extend([
            "",
            "[Tuning]",
            "ball_area_ratio min=" + str(MIN_BALL_AREA_RATIO)
            + " top_scale=" + str(MIN_BALL_AREA_TOP_SCALE)
            + " max=" + str(MAX_BALL_AREA_RATIO)
            + " close=" + str(CLOSE_BALL_AREA_RATIO),
            "shape circularity_min=" + str(MIN_BALL_CIRCULARITY)
            + " circle_fill_min=" + str(MIN_BALL_CIRCLE_FILL)
            + " triangle_epsilon=" + str(TRIANGLE_APPROX_EPSILON),
            "known_color hue_padding=" + str(KNOWN_COLOR_HUE_PADDING)
            + " sat_min=" + str(KNOWN_COLOR_SATURATION_MIN)
            + " value_min=" + str(KNOWN_COLOR_VALUE_MIN),
            "drive gain=" + str(STEERING_GAIN)
            + " deadband=" + str(STEERING_DEADBAND)
            + " max_throttle=" + str(MAX_TRIAL_THROTTLE),
            "steering degrees L/C/R="
            + str(STEERING_LEFT_DEGREES)
            + "/"
            + str(STEERING_CENTER_DEGREES)
            + "/"
            + str(STEERING_RIGHT_DEGREES),
            "esc neutral/forward/reverse us="
            + str(THROTTLE_NEUTRAL_US)
            + "/"
            + str(THROTTLE_FORWARD_US)
            + "/"
            + str(THROTTLE_REVERSE_US),
            "esc arm_seconds=" + str(ESC_ARM_SECONDS)
            + " hard_limit=" + str(THROTTLE_HARD_LIMIT)
            + " min_active=" + str(THROTTLE_MIN_ACTIVE),
            "priority weights distance=" + str(TARGET_DISTANCE_WEIGHT)
            + " cluster=" + str(TARGET_CLUSTER_WEIGHT)
            + " area=" + str(TARGET_AREA_WEIGHT)
            + " center=" + str(TARGET_CENTER_WEIGHT),
            "priority cluster_radius_ratio=" + str(TARGET_CLUSTER_RADIUS_RATIO),
            "target lock_radius_ratio=" + str(TARGET_LOCK_RADIUS_RATIO)
            + " switch_margin=" + str(TARGET_SWITCH_MARGIN)
            + " switch_frames=" + str(TARGET_SWITCH_FRAMES),
            "target hold_seconds=" + str(TARGET_HOLD_SECONDS)
            + " smoothing=" + str(TARGET_SMOOTHING)
            + " steering_slew_rate=" + str(STEERING_SLEW_RATE),
        ])

        return lines

    def make_steering_bar(self, steering, terminal_width):
        label = "steering L "
        suffix = " R"
        bar_width = max(11, terminal_width - len(label) - len(suffix) - 1)
        center_index = bar_width // 2
        marker_index = int(round((clamp(steering, -1.0, 1.0) + 1.0) * 0.5 * (bar_width - 1)))
        bar = ["-"] * bar_width
        bar[center_index] = "|"
        bar[marker_index] = "#"

        return label + "".join(bar) + suffix


def tui_is_active():
    dashboard = globals().get("dashboard")
    return dashboard is not None and dashboard.enabled


def hue_distance(first_hue, second_hue):
    direct_distance = abs(int(first_hue) - int(second_hue))
    return min(direct_distance, 180 - direct_distance)


def make_range_from_hsv_pixels(hsv_pixels):
    hue_values = hsv_pixels[:, 0].astype(int)
    saturation_values = hsv_pixels[:, 1].astype(int)
    value_values = hsv_pixels[:, 2].astype(int)
    hue = int(np.median(hue_values))
    saturation = int(np.median(saturation_values))
    value = int(np.median(value_values))

    return make_range_from_hsv_with_padding(
        (hue, saturation, value),
        AUTO_CALIBRATION_HUE_PADDING,
        AUTO_CALIBRATION_SATURATION_PADDING,
        AUTO_CALIBRATION_VALUE_PADDING
    )


def make_range_from_hsv_with_padding(hsv_value, hue_padding, saturation_padding, value_padding):
    hue = int(hsv_value[0])
    saturation = int(hsv_value[1])
    value = int(hsv_value[2])

    lower_hue = hue - hue_padding
    upper_hue = hue + hue_padding
    lower_saturation = max(0, saturation - saturation_padding)
    upper_saturation = min(255, saturation + saturation_padding)
    lower_value = max(0, value - value_padding)
    upper_value = min(255, value + value_padding)

    if lower_hue < 0:
        return {
            "lower": np.array([0, lower_saturation, lower_value]),
            "upper": np.array([upper_hue, upper_saturation, upper_value]),
            "lower2": np.array([179 + lower_hue, lower_saturation, lower_value]),
            "upper2": np.array([179, upper_saturation, upper_value])
        }

    if upper_hue > 179:
        return {
            "lower": np.array([lower_hue, lower_saturation, lower_value]),
            "upper": np.array([179, upper_saturation, upper_value]),
            "lower2": np.array([0, lower_saturation, lower_value]),
            "upper2": np.array([upper_hue - 179, upper_saturation, upper_value])
        }

    return {
        "lower": np.array([lower_hue, lower_saturation, lower_value]),
        "upper": np.array([upper_hue, upper_saturation, upper_value])
    }


KNOWN_BALL_COLOR_SPECS = [
    ("red", 0, (0, 0, 255)),
    ("orange", 10, (0, 140, 255)),
    ("yellow", 30, (0, 255, 255)),
    ("green", 60, (0, 255, 0)),
    ("blue", 110, (255, 0, 0)),
    ("purple", 135, (255, 0, 180)),
    ("pink", 160, (255, 0, 255)),
]


def make_known_ball_colors():
    known_colors = []

    for name, hue, box_color in KNOWN_BALL_COLOR_SPECS:
        color_range = make_range_from_hsv_with_padding(
            (hue, 150, 160),
            KNOWN_COLOR_HUE_PADDING,
            255 - KNOWN_COLOR_SATURATION_MIN,
            255 - KNOWN_COLOR_VALUE_MIN
        )
        color_range["lower"][2] = max(
            int(color_range["lower"][2]),
            KNOWN_COLOR_VALUE_MIN
        )

        if "lower2" in color_range:
            color_range["lower2"][2] = max(
                int(color_range["lower2"][2]),
                KNOWN_COLOR_VALUE_MIN
            )

        color = {
            "name": name,
            "lower": color_range["lower"],
            "upper": color_range["upper"],
            "box_color": box_color,
            "known_ball_color": True,
            "min_detection_saturation": KNOWN_COLOR_SATURATION_MIN
        }

        if "lower2" in color_range:
            color["lower2"] = color_range["lower2"]
            color["upper2"] = color_range["upper2"]

        known_colors.append(color)

    return known_colors


KNOWN_BALL_COLORS = make_known_ball_colors()


def classify_known_ball_hue(hue):
    closest_name = None
    closest_distance = 999

    for name, known_hue, _ in KNOWN_BALL_COLOR_SPECS:
        distance = hue_distance(hue, known_hue)

        if distance < closest_distance:
            closest_name = name
            closest_distance = distance

    if closest_distance <= KNOWN_COLOR_HUE_PADDING + 4:
        return closest_name

    return None


class AutoCalibrator:
    def __init__(self):
        self.profiles = []
        self.last_update_time = 0
        self.last_candidate_count = 0

    def update(self, frame, hsv, current_time):
        if not AUTO_CALIBRATE:
            return []

        if current_time - self.last_update_time < AUTO_CALIBRATION_INTERVAL:
            return self.as_colors()

        self.last_update_time = current_time
        candidates = self.find_candidates(frame, hsv)
        self.last_candidate_count = len(candidates)

        for candidate in candidates:
            self.merge_candidate(candidate, current_time)

        return self.as_colors()

    def find_candidates(self, frame, hsv):
        minimum_area = frame.shape[0] * frame.shape[1] * AUTO_CALIBRATION_MIN_AREA_RATIO
        saturated_mask = cv2.inRange(
            hsv,
            np.array([
                0,
                AUTO_CALIBRATION_SATURATION_MIN,
                AUTO_CALIBRATION_VALUE_MIN
            ]),
            np.array([179, 255, 255])
        )
        kernel = np.ones((7, 7), np.uint8)
        saturated_mask = cv2.morphologyEx(saturated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        saturated_mask = cv2.erode(saturated_mask, None, iterations=1)
        saturated_mask = cv2.dilate(saturated_mask, None, iterations=2)
        contours, _ = cv2.findContours(
            saturated_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < minimum_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if not is_spherical(contour, w, h, saturated_mask):
                continue

            contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            pixels = hsv[contour_mask == 255]

            if len(pixels) < 30:
                continue

            hue = int(np.median(pixels[:, 0]))
            new_range = make_range_from_hsv_pixels(pixels)
            box_color = make_box_color_from_hsv((
                hue,
                int(np.median(pixels[:, 1])),
                int(np.median(pixels[:, 2]))
            ))
            classified_name = classify_known_ball_hue(hue)

            if classified_name is None:
                continue

            candidates.append(
                AutoCalibrationProfile(
                    name=classified_name + "-auto",
                    hue=hue,
                    color_range=new_range,
                    box_color=box_color,
                    last_seen_time=0,
                    observations=1
                )
            )

        return candidates

    def merge_candidate(self, candidate, current_time):
        for profile in self.profiles:
            if hue_distance(profile.hue, candidate.hue) <= AUTO_CALIBRATION_MERGE_HUE_DISTANCE:
                observations = profile.observations + 1
                profile.hue = int(
                    (
                        profile.hue * profile.observations
                        + candidate.hue
                    )
                    / observations
                )
                profile.color_range = candidate.color_range
                profile.box_color = candidate.box_color
                profile.name = candidate.name
                profile.last_seen_time = current_time
                profile.observations = observations
                return

        candidate.last_seen_time = current_time
        self.profiles.append(candidate)
        self.profiles = sorted(
            self.profiles,
            key=lambda profile: profile.last_seen_time,
            reverse=True
        )[:AUTO_CALIBRATION_MAX_COLORS]
        print(
            "Auto-calibrated color:",
            candidate.name,
            "hue=" + str(candidate.hue),
            "range=" + array_to_code(candidate.color_range["lower"]),
            "to",
            array_to_code(candidate.color_range["upper"])
        )

    def as_colors(self):
        auto_colors = []

        for profile in self.profiles:
            color = {
                "name": profile.name,
                "lower": profile.color_range["lower"],
                "upper": profile.color_range["upper"],
                "box_color": profile.box_color,
                "auto_calibrated": True
            }

            if "lower2" in profile.color_range:
                color["lower2"] = profile.color_range["lower2"]
                color["upper2"] = profile.color_range["upper2"]

            auto_colors.append(color)

        return auto_colors


def list_video_devices():
    try:
        return sorted(
            device
            for device in os.listdir("/dev")
            if device.startswith("video")
        )
    except OSError as error:
        return ["could not list /dev: " + str(error)]


def print_camera_debug_header():
    print("Camera debug:")
    print("  backend request:", CAMERA_BACKEND)
    print("  headless:", HEADLESS)
    print("  actuators enabled:", ENABLE_ACTUATORS)
    print("  auto calibration:", AUTO_CALIBRATE)
    print("  tui requested:", TUI_ENABLED)
    print("  frame size:", str(FRAME_WIDTH) + "x" + str(FRAME_HEIGHT))
    print("  python:", platform.python_version())
    print("  platform:", platform.platform())
    print("  opencv:", cv2.__version__)
    print("  /dev video devices:", ", ".join(list_video_devices()) or "none")
    print()


class Pca9685Actuators:
    def __init__(self):
        self.enabled = ENABLE_ACTUATORS
        self.last_command = None
        self.esc_armed = False
        self.last_steering_us = None
        self.last_throttle_us = None

        if not self.enabled:
            print("Actuators disabled. Set ENABLE_ACTUATORS=true to drive PCA9685 outputs.")
            return

        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError:
            print("PCA9685 libraries are missing. Install with:")
            print("  sudo apt install -y python3-pip")
            print("  pip3 install adafruit-circuitpython-pca9685")
            raise

        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50
        print("PCA9685 actuator output enabled.")
        print("  steering channel:", STEERING_CHANNEL)
        print("  steering degrees left/center/right:", STEERING_LEFT_DEGREES, STEERING_CENTER_DEGREES, STEERING_RIGHT_DEGREES)
        print("  throttle channel:", THROTTLE_CHANNEL)
        print("  throttle neutral/forward/reverse us:", THROTTLE_NEUTRAL_US, THROTTLE_FORWARD_US, THROTTLE_REVERSE_US)
        print("  throttle hard limit:", THROTTLE_HARD_LIMIT)
        print("  throttle minimum active:", THROTTLE_MIN_ACTIVE)
        print("  throttle movement enabled:", ENABLE_THROTTLE)
        print("  holding neutral for ESC arm seconds:", ESC_ARM_SECONDS)
        self.neutralize()
        time.sleep(ESC_ARM_SECONDS)
        self.esc_armed = True

    def set_pulse_us(self, channel, pulse_us):
        if not self.enabled:
            return

        duty_cycle = int(clamp(pulse_us * 50 * 65535 / 1000000, 0, 65535))
        self.pca.channels[channel].duty_cycle = duty_cycle

    def apply(self, command):
        steering_us = normalized_to_steering_pulse(command.steering)
        throttle_us = normalized_to_esc_throttle_pulse(command.throttle)
        safe_throttle = normalize_throttle_for_esc(command.throttle)
        command_state = (
            round(command.steering, 3),
            round(safe_throttle, 3),
            command.mode,
            command.reason,
            steering_us,
            throttle_us
        )
        self.last_steering_us = steering_us
        self.last_throttle_us = throttle_us

        if command_state != self.last_command and ENABLE_ACTUATORS and not tui_is_active():
            print(
                "Drive command:",
                "mode=" + command.mode,
                "steering=" + str(round(command.steering, 3)),
                "requested_throttle=" + str(round(command.throttle, 3)),
                "safe_throttle=" + str(round(safe_throttle, 3)),
                "steering_us=" + str(steering_us),
                "throttle_us=" + str(throttle_us),
                "reason=" + command.reason
            )
            self.last_command = command_state

        self.set_pulse_us(STEERING_CHANNEL, steering_us)
        self.set_pulse_us(THROTTLE_CHANNEL, throttle_us)

    def neutralize(self):
        self.apply(DriveCommand(0.0, 0.0, "disabled", "neutralize"))

    def close(self):
        self.neutralize()

        if self.enabled:
            self.pca.deinit()


class Picamera2Camera:
    def __init__(self, width, height):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={
                "size": (width, height),
                "format": "BGR888"
            }
        )
        self.camera.configure(config)
        self.camera.start()
        self.frame_count = 0
        print("Picamera2 started.")
        print("  configured size:", str(width) + "x" + str(height))
        print("  configured format: BGR888")
        print("  swap red/blue:", SWAP_PICAMERA2_RED_BLUE)

    def read(self):
        try:
            frame = self.camera.capture_array()
        except Exception:
            print("Picamera2 capture_array() failed:")
            traceback.print_exc()
            return False, None

        if frame is None:
            print("Picamera2 returned no frame.")
            return False, None

        if len(frame.shape) != 3 or frame.shape[2] < 3:
            print("Picamera2 returned an unexpected frame shape:", frame.shape)
            return False, None

        if self.frame_count == 0:
            print("First Picamera2 frame shape:", frame.shape)

        self.frame_count += 1
        frame = frame[:, :, :3]

        if SWAP_PICAMERA2_RED_BLUE:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return True, frame

    def release(self):
        self.camera.stop()


def open_camera(width, height):
    if CAMERA_BACKEND not in VALID_CAMERA_BACKENDS:
        raise ValueError(
            "CAMERA_BACKEND must be one of: "
            + ", ".join(VALID_CAMERA_BACKENDS)
        )

    print_camera_debug_header()

    if CAMERA_BACKEND in ("auto", "picamera2"):
        try:
            print("Opening camera with Picamera2/libcamera.")
            return Picamera2Camera(width, height)
        except ImportError:
            print("Picamera2 import failed:")
            traceback.print_exc()
            print()

            if CAMERA_BACKEND == "picamera2" or (
                CAMERA_BACKEND == "auto" and os.name != "nt"
            ):
                raise

            print("Falling back to OpenCV VideoCapture.")
        except Exception:
            print("Picamera2 startup failed:")
            traceback.print_exc()
            print()

            if CAMERA_BACKEND == "picamera2" or (
                CAMERA_BACKEND == "auto" and os.name != "nt"
            ):
                raise

            print("Falling back to OpenCV VideoCapture.")

    print("Opening camera with OpenCV VideoCapture(0).")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("OpenCV camera opened:", camera.isOpened())
    print("  requested size:", str(width) + "x" + str(height))
    print("  reported width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("  reported height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("  reported backend:", camera.getBackendName() if camera.isOpened() else "none")

    return camera


actuators = Pca9685Actuators()

camera = open_camera(FRAME_WIDTH, FRAME_HEIGHT)
auto_calibrator = AutoCalibrator()
dashboard = TuiDashboard()
window_name = "Tennis Ball Detector"
last_hsv_sample = None
last_click = None
last_assignment = "No color assigned yet"
current_hsv = None
last_detection_print_time = 0
last_target_seen_time = 0
last_frame_time = 0
current_fps = 0
naming_mode = False
typed_color_name = ""
delete_mode = False
typed_delete_name = ""
calibration_samples = []
samples_needed = 3
roi_mode = False
roi_start = None
roi_end = None
roi_box = None


def print_camera_read_failure(camera):
    print("Could not access camera")
    print("Camera read failure debug:")
    print("  selected backend:", type(camera).__name__)

    if hasattr(camera, "isOpened"):
        print("  opencv isOpened:", camera.isOpened())
        print("  opencv backend:", camera.getBackendName() if camera.isOpened() else "none")
        print("  opencv width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("  opencv height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("  /dev video devices:", ", ".join(list_video_devices()) or "none")
    print()
    print("Try these on the Pi for more context:")
    print("  CAMERA_BACKEND=picamera2 python3 detector.py")
    print("  python3 -c \"from picamera2 import Picamera2; print(Picamera2.global_camera_info())\"")
    print("  libcamera-hello --list-cameras")
    print()


def make_range_from_hsv(hsv_value):
    hue_range = 5
    saturation_range = 40
    value_range = 40

    hue = int(hsv_value[0])
    saturation = int(hsv_value[1])
    value = int(hsv_value[2])

    lower_hue = hue - hue_range
    upper_hue = hue + hue_range

    lower_saturation = max(0, saturation - saturation_range)
    upper_saturation = min(255, saturation + saturation_range)
    lower_value = max(0, value - value_range)
    upper_value = min(255, value + value_range)

    if lower_hue < 0:
        return {
            "lower": np.array([0, lower_saturation, lower_value]),
            "upper": np.array([upper_hue, upper_saturation, upper_value]),
            "lower2": np.array([179 + lower_hue, lower_saturation, lower_value]),
            "upper2": np.array([179, upper_saturation, upper_value])
        }

    if upper_hue > 179:
        return {
            "lower": np.array([lower_hue, lower_saturation, lower_value]),
            "upper": np.array([179, upper_saturation, upper_value]),
            "lower2": np.array([0, lower_saturation, lower_value]),
            "upper2": np.array([upper_hue - 179, upper_saturation, upper_value])
        }

    return {
        "lower": np.array([lower_hue, lower_saturation, lower_value]),
        "upper": np.array([upper_hue, upper_saturation, upper_value])
    }


def make_range_from_samples(samples):
    hue_padding = 4
    saturation_padding = 25
    value_padding = 25

    sample_array = np.array(samples)

    lower = np.array([
        max(0, int(np.min(sample_array[:, 0])) - hue_padding),
        max(0, int(np.min(sample_array[:, 1])) - saturation_padding),
        max(0, int(np.min(sample_array[:, 2])) - value_padding)
    ])

    upper = np.array([
        min(179, int(np.max(sample_array[:, 0])) + hue_padding),
        min(255, int(np.max(sample_array[:, 1])) + saturation_padding),
        min(255, int(np.max(sample_array[:, 2])) + value_padding)
    ])

    return {
        "lower": lower,
        "upper": upper
    }


def mask_from_range(hsv, color_range):
    mask = cv2.inRange(
        hsv,
        color_range["lower"],
        color_range["upper"]
    )

    if "lower2" in color_range:
        mask2 = cv2.inRange(
            hsv,
            color_range["lower2"],
            color_range["upper2"]
        )
        mask = cv2.bitwise_or(mask, mask2)

    return mask


def range_with_saturation_floor(color_range, saturation_floor):
    adjusted_range = color_range.copy()
    adjusted_range["lower"] = color_range["lower"].copy()
    adjusted_range["lower"][1] = max(
        int(adjusted_range["lower"][1]),
        saturation_floor
    )

    if "lower2" in color_range:
        adjusted_range["lower2"] = color_range["lower2"].copy()
        adjusted_range["lower2"][1] = max(
            int(adjusted_range["lower2"][1]),
            saturation_floor
        )

    return adjusted_range


def print_color_range(color):
    print(color["name"] + " range:")
    print(
        '"lower": np.array(['
        + str(color["lower"][0])
        + ", "
        + str(color["lower"][1])
        + ", "
        + str(color["lower"][2])
        + "]),"
    )
    print(
        '"upper": np.array(['
        + str(color["upper"][0])
        + ", "
        + str(color["upper"][1])
        + ", "
        + str(color["upper"][2])
        + "]),"
    )

    if "lower2" in color:
        print(
            '"lower2": np.array(['
            + str(color["lower2"][0])
            + ", "
            + str(color["lower2"][1])
            + ", "
            + str(color["lower2"][2])
            + "]),"
        )
        print(
            '"upper2": np.array(['
            + str(color["upper2"][0])
            + ", "
            + str(color["upper2"][1])
            + ", "
            + str(color["upper2"][2])
            + "]),"
        )

    print()


def array_to_code(array):
    return (
        "np.array(["
        + str(int(array[0]))
        + ", "
        + str(int(array[1]))
        + ", "
        + str(int(array[2]))
        + "])"
    )


def print_hardcode_color(color):
    print("Copy this into the starting colors list:")
    print("{")
    print('    "name": "' + color["name"] + '",')
    print('    "lower": ' + array_to_code(color["lower"]) + ",")
    print('    "upper": ' + array_to_code(color["upper"]) + ",")

    if "lower2" in color:
        print('    "lower2": ' + array_to_code(color["lower2"]) + ",")
        print('    "upper2": ' + array_to_code(color["upper2"]) + ",")

    if "sample_ranges" in color:
        print('    "sample_ranges": [')

        for sample_range in color["sample_ranges"]:
            print("        {")
            print('            "lower": ' + array_to_code(sample_range["lower"]) + ",")
            print('            "upper": ' + array_to_code(sample_range["upper"]) + ",")

            if "lower2" in sample_range:
                print(
                    '            "lower2": '
                    + array_to_code(sample_range["lower2"])
                    + ","
                )
                print(
                    '            "upper2": '
                    + array_to_code(sample_range["upper2"])
                    + ","
                )

            print("        },")

        print("    ],")

    print(
        '    "box_color": ('
        + str(int(color["box_color"][0]))
        + ", "
        + str(int(color["box_color"][1]))
        + ", "
        + str(int(color["box_color"][2]))
        + ")"
    )
    print("},")
    print()


def print_all_hardcode_colors():
    print("Current full hard-code colors list:")
    print("colors = [")

    for color in colors:
        print("    {")
        print('        "name": "' + color["name"] + '",')
        print('        "lower": ' + array_to_code(color["lower"]) + ",")
        print('        "upper": ' + array_to_code(color["upper"]) + ",")

        if "lower2" in color:
            print('        "lower2": ' + array_to_code(color["lower2"]) + ",")
            print('        "upper2": ' + array_to_code(color["upper2"]) + ",")

        if "sample_ranges" in color:
            print('        "sample_ranges": [')

            for sample_range in color["sample_ranges"]:
                print("            {")
                print(
                    '                "lower": '
                    + array_to_code(sample_range["lower"])
                    + ","
                )
                print(
                    '                "upper": '
                    + array_to_code(sample_range["upper"])
                    + ","
                )

                if "lower2" in sample_range:
                    print(
                        '                "lower2": '
                        + array_to_code(sample_range["lower2"])
                        + ","
                    )
                    print(
                        '                "upper2": '
                        + array_to_code(sample_range["upper2"])
                        + ","
                    )

                print("            },")

            print("        ],")

        print(
            '        "box_color": ('
            + str(int(color["box_color"][0]))
            + ", "
            + str(int(color["box_color"][1]))
            + ", "
            + str(int(color["box_color"][2]))
            + ")"
        )
        print("    },")

    print("]")
    print()


def make_box_color_from_hsv(hsv_value):
    hsv_pixel = np.uint8([[[hsv_value[0], hsv_value[1], hsv_value[2]]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]

    return (
        int(bgr_pixel[0]),
        int(bgr_pixel[1]),
        int(bgr_pixel[2])
    )


def add_color_from_last_sample(color_name):
    global last_assignment

    if len(calibration_samples) < samples_needed:
        last_assignment = (
            "Click "
            + str(samples_needed - len(calibration_samples))
            + " more ball area(s) before adding a color."
        )
        print(last_assignment)
        return

    color_name = color_name.strip()

    if color_name == "":
        last_assignment = "No color name entered."
        print(last_assignment)
        return

    average_sample = np.mean(np.array(calibration_samples), axis=0).astype(int)
    new_range = make_range_from_samples(calibration_samples)
    color = {
        "name": color_name,
        "lower": new_range["lower"],
        "upper": new_range["upper"],
        "sample_ranges": [
            make_range_from_hsv(sample)
            for sample in calibration_samples
        ],
        "box_color": make_box_color_from_hsv(average_sample)
    }

    if "lower2" in new_range:
        color["lower2"] = new_range["lower2"]
        color["upper2"] = new_range["upper2"]

    colors.append(color)
    last_assignment = "Added color: " + color_name
    calibration_samples.clear()

    print(last_assignment)
    print_color_range(color)
    print_hardcode_color(color)
    print_all_hardcode_colors()


def delete_color_by_name(color_name):
    global last_assignment

    color_name = color_name.strip().lower()

    if color_name == "":
        last_assignment = "No color name entered."
        return

    for color in colors:
        if color["name"].lower() == color_name:
            colors.remove(color)
            last_assignment = "Deleted color: " + color["name"]
            print(last_assignment)
            return

    last_assignment = "Could not find color: " + color_name
    print(last_assignment)


def handle_key(key):
    global naming_mode, typed_color_name, delete_mode, typed_delete_name
    global last_assignment, roi_mode, roi_start, roi_end, roi_box

    enter_pressed = key == 10 or key == 13
    backspace_pressed = key == 8 or key == 127
    escape_pressed = key == 27

    if key == ord("c") or key == ord("C"):
        calibration_samples.clear()
        typed_color_name = ""
        naming_mode = False
        typed_delete_name = ""
        delete_mode = False
        roi_mode = False
        roi_start = None
        roi_end = None
        last_assignment = "Cleared samples and returned to normal mode."
        return

    if escape_pressed:
        typed_color_name = ""
        naming_mode = False
        typed_delete_name = ""
        delete_mode = False
        roi_mode = False
        roi_start = None
        roi_end = None
        last_assignment = "Canceled and returned to normal mode."
        return

    if delete_mode:
        if enter_pressed:
            delete_color_by_name(typed_delete_name)
            typed_delete_name = ""
            delete_mode = False
            return

        if backspace_pressed:
            typed_delete_name = typed_delete_name[:-1]
            return

        if 32 <= key <= 126:
            typed_delete_name += chr(key)
            return

    if naming_mode:
        if enter_pressed:
            add_color_from_last_sample(typed_color_name)
            typed_color_name = ""
            naming_mode = False
            return

        if backspace_pressed:
            typed_color_name = typed_color_name[:-1]
            return

        if 32 <= key <= 126:
            typed_color_name += chr(key)
            return

    if key == ord("a") or key == ord("A"):
        if len(calibration_samples) < samples_needed:
            last_assignment = (
                "Click "
                + str(samples_needed - len(calibration_samples))
                + " more ball area(s) before naming."
            )
            print(last_assignment)
            return

        naming_mode = True
        typed_color_name = ""
        last_assignment = "Type color name, then press Enter."

    if key == ord("d") or key == ord("D"):
        delete_mode = True
        typed_delete_name = ""
        naming_mode = False
        typed_color_name = ""
        last_assignment = "Type color name to delete, then press Enter."

    if key == ord("r") or key == ord("R"):
        roi_mode = True
        roi_start = None
        roi_end = None
        naming_mode = False
        delete_mode = False
        last_assignment = "Drag a box around the ball play area."
        return

    if key == ord("x") or key == ord("X"):
        roi_box = None
        roi_start = None
        roi_end = None
        roi_mode = False
        last_assignment = "Detection area removed."
        return



def show_hsv_value(event, x, y, flags, param):
    global last_hsv_sample, last_click, last_assignment
    global roi_start, roi_end, roi_box, roi_mode

    if roi_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_start = (x, y)
            roi_end = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and roi_start is not None:
            roi_end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and roi_start is not None:
            x1 = min(roi_start[0], x)
            y1 = min(roi_start[1], y)
            x2 = max(roi_start[0], x)
            y2 = max(roi_start[1], y)

            if x2 - x1 > 20 and y2 - y1 > 20:
                roi_box = (x1, y1, x2, y2)
                last_assignment = "Detection area set."
                print(last_assignment)
            else:
                last_assignment = "Detection area was too small."
                print(last_assignment)

            roi_start = None
            roi_end = None
            roi_mode = False
            return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if naming_mode or delete_mode:
        return

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box

        if x < x1 or x > x2 or y < y1 or y > y2:
            last_assignment = "Click inside the detection area."
            return

    if current_hsv is None:
        return

    hsv = current_hsv
    height, width = hsv.shape[:2]

    sample_size = 5
    x1 = max(0, x - sample_size)
    x2 = min(width, x + sample_size + 1)
    y1 = max(0, y - sample_size)
    y2 = min(height, y + sample_size + 1)

    sample_area = hsv[y1:y2, x1:x2]
    average_hsv = np.mean(sample_area.reshape(-1, 3), axis=0).astype(int)

    last_hsv_sample = average_hsv
    last_click = (x, y)
    calibration_samples.append(average_hsv)

    if len(calibration_samples) > samples_needed:
        calibration_samples.pop(0)

    print(
        "Clicked HSV:",
        "H =", average_hsv[0],
        "S =", average_hsv[1],
        "V =", average_hsv[2]
    )

    samples_left = samples_needed - len(calibration_samples)

    if samples_left > 0:
        last_assignment = (
            "Sample "
            + str(len(calibration_samples))
            + "/"
            + str(samples_needed)
            + " saved. Click "
            + str(samples_left)
            + " more area(s)."
        )
        print(last_assignment)
    else:
        last_assignment = "3 samples saved. Press A to name this color."
        print(last_assignment)

    print("Click 3 ball areas, then press A and type the color name.")
    print()


if not HEADLESS:
    cv2.namedWindow(window_name)
    print("Visualization window enabled. Runtime mouse/key calibration is disabled.")
else:
    print("Running headless: camera windows and mouse calibration are disabled.")


# Starting colors from your calibration.
# You can still add or delete colors while the program is running.
colors = [
    {
        "name": "yellow",
        "lower": np.array([33, 0, 167]),
        "upper": np.array([54, 130, 255]),
        "sample_ranges": [
            {
                "lower": np.array([45, 0, 214]),
                "upper": np.array([55, 52, 255]),
            },
            {
                "lower": np.array([33, 65, 215]),
                "upper": np.array([43, 145, 255]),
            },
            {
                "lower": np.array([32, 60, 152]),
                "upper": np.array([42, 140, 232]),
            },
        ],
        "box_color": (167, 233, 209),
        "min_detection_saturation": 45
    },
    {
        "name": "pink",
        "lower": np.array([147, 27, 160]),
        "upper": np.array([167, 153, 255]),
        "sample_ranges": [
            {
                "lower": np.array([146, 12, 215]),
                "upper": np.array([156, 92, 255]),
            },
            {
                "lower": np.array([158, 88, 207]),
                "upper": np.array([168, 168, 255]),
            },
            {
                "lower": np.array([155, 54, 145]),
                "upper": np.array([165, 134, 225]),
            },
        ],
        "box_color": (207, 147, 229)
    },
    {
        "name": "blue",
        "lower": np.array([101, 190, 102]),
        "upper": np.array([113, 255, 252]),
        "sample_ranges": [
            {
                "lower": np.array([100, 175, 187]),
                "upper": np.array([110, 255, 255]),
            },
            {
                "lower": np.array([102, 209, 151]),
                "upper": np.array([112, 255, 231]),
            },
            {
                "lower": np.array([104, 183, 87]),
                "upper": np.array([114, 255, 167]),
            },
        ],
        "box_color": (181, 89, 18)
    },
    {
        "name": "lime",
        "lower": np.array([62, 0, 165]),
        "upper": np.array([83, 136, 255]),
        "sample_ranges": [
            {
                "lower": np.array([74, 0, 214]),
                "upper": np.array([84, 64, 255]),
            },
            {
                "lower": np.array([62, 71, 203]),
                "upper": np.array([72, 151, 255]),
            },
            {
                "lower": np.array([61, 53, 150]),
                "upper": np.array([71, 133, 230]),
            },
        ],
        "box_color": (183, 229, 161),
        "min_detection_saturation": 45
    },
    {
        "name": "purple",
        "lower": np.array([121, 97, 96]),
        "upper": np.array([130, 205, 248]),
        "sample_ranges": [
            {
                "lower": np.array([121, 82, 183]),
                "upper": np.array([131, 162, 255]),
            },
            {
                "lower": np.array([120, 140, 131]),
                "upper": np.array([130, 220, 211]),
            },
            {
                "lower": np.array([120, 107, 81]),
                "upper": np.array([130, 187, 161]),
            },
        ],
        "box_color": (171, 71, 88)
    },
    {
        "name": "red",
        "lower": np.array([168, 137, 141]),
        "upper": np.array([177, 231, 255]),
        "sample_ranges": [
            {
                "lower": np.array([168, 123, 214]),
                "upper": np.array([178, 203, 255]),
            },
            {
                "lower": np.array([168, 166, 170]),
                "upper": np.array([178, 246, 250]),
            },
            {
                "lower": np.array([167, 122, 126]),
                "upper": np.array([177, 202, 206]),
            },
        ],
        "box_color": (103, 64, 210)
    },
    {
        "name": "orange",
        "lower": np.array([0, 80, 181]),
        "upper": np.array([13, 187, 255]),
        "sample_ranges": [
            {
                "lower": np.array([4, 65, 215]),
                "upper": np.array([14, 145, 255]),
            },
            {
                "lower": np.array([0, 105, 215]),
                "upper": np.array([6, 185, 255]),
                "lower2": np.array([175, 105, 215]),
                "upper2": np.array([179, 185, 255]),
            },
            {
                "lower": np.array([0, 122, 166]),
                "upper": np.array([8, 202, 246]),
                "lower2": np.array([177, 122, 166]),
                "upper2": np.array([179, 202, 246]),
            },
        ],
        "box_color": (110, 127, 238)
    },
    {
        "name": "lightblue",
        "lower": np.array([91, 97, 155]),
        "upper": np.array([104, 220, 255]),
        "sample_ranges": [
            {
                "lower": np.array([90, 82, 214]),
                "upper": np.array([100, 162, 255]),
            },
            {
                "lower": np.array([94, 155, 199]),
                "upper": np.array([104, 235, 255]),
            },
            {
                "lower": np.array([95, 152, 140]),
                "upper": np.array([105, 232, 220]),
            },
        ],
        "box_color": (224, 184, 76)
    },
]


def make_mask(hsv, color):
    detection_range = color

    if "min_detection_saturation" in color:
        detection_range = range_with_saturation_floor(
            color,
            color["min_detection_saturation"]
        )

    mask = mask_from_range(hsv, detection_range)

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def is_known_or_auto_color(color):
    return color.get("known_ball_color", False) or color.get("auto_calibrated", False)


def contour_is_triangular(contour, perimeter):
    epsilon = max(0.001, TRIANGLE_APPROX_EPSILON) * perimeter
    approximated = cv2.approxPolyDP(contour, epsilon, True)

    return len(approximated) <= 4 and cv2.isContourConvex(approximated)


def is_spherical(contour, width, height, mask):
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False

    if contour_is_triangular(contour, perimeter):
        return False

    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    width_height_ratio = width / float(height)
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)

    if radius == 0:
        return False

    circle_area = np.pi * radius * radius
    circle_fill_ratio = area / circle_area
    center_x = int(circle_x)
    center_y = int(circle_y)
    circle_mask = np.zeros_like(mask)

    cv2.circle(
        circle_mask,
        (center_x, center_y),
        int(radius),
        255,
        -1
    )

    matching_pixels = cv2.countNonZero(
        cv2.bitwise_and(mask, circle_mask)
    )
    circle_pixels = cv2.countNonZero(circle_mask)

    if circle_pixels == 0:
        return False

    color_fill_ratio = matching_pixels / float(circle_pixels)

    if width_height_ratio < 0.6 or width_height_ratio > 1.4:
        return False

    if circularity < MIN_BALL_CIRCULARITY:
        return False

    if circle_fill_ratio < MIN_BALL_CIRCLE_FILL:
        return False

    if color_fill_ratio < 0.45:
        return False

    return True


def has_sampled_color_variety(contour, hsv, color):
    if "sample_ranges" not in color:
        return True

    contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    cv2.drawContours(
        contour_mask,
        [contour],
        -1,
        255,
        -1
    )

    matching_sample_count = 0
    contour_area = cv2.countNonZero(contour_mask)

    for sample_range in color["sample_ranges"]:
        sample_mask = mask_from_range(hsv, sample_range)
        sample_mask = cv2.bitwise_and(sample_mask, contour_mask)
        matching_pixels = cv2.countNonZero(sample_mask)

        if matching_pixels > max(30, contour_area * 0.08):
            matching_sample_count += 1

    needed_matches = min(2, len(color["sample_ranges"]))

    return matching_sample_count >= needed_matches


def box_area(box):
    return max(0, box[2]) * max(0, box[3])


def box_intersection_area(first_box, second_box):
    first_x, first_y, first_w, first_h = first_box
    second_x, second_y, second_w, second_h = second_box

    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)

    return max(0, right - left) * max(0, bottom - top)


def boxes_nearly_duplicate(first_box, second_box):
    intersection = box_intersection_area(first_box, second_box)
    if intersection == 0:
        return False

    smaller_area = min(box_area(first_box), box_area(second_box))
    if smaller_area <= 0:
        return False

    return intersection / float(smaller_area) >= 0.85


def cull_overlapping_targets(targets):
    kept_targets = []
    sorted_targets = sorted(
        targets,
        key=lambda target: target.area,
        reverse=True
    )

    for target in sorted_targets:
        if any(boxes_nearly_duplicate(target.box, kept_target.box) for kept_target in kept_targets):
            continue

        kept_targets.append(target)

    return kept_targets


def minimum_ball_area_for_y(center_y, frame_height, frame_area):
    y_ratio = clamp(center_y / float(max(1, frame_height)), 0.0, 1.0)
    top_scale = clamp(MIN_BALL_AREA_TOP_SCALE, 0.0, 1.0)
    area_ratio = MIN_BALL_AREA_RATIO * (top_scale + (1.0 - top_scale) * y_ratio)
    return frame_area * area_ratio


def detect_tennis_balls(frame, hsv, active_colors):
    targets = []
    detected_ball_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    frame_area = frame.shape[0] * frame.shape[1]
    maximum_ball_area = frame_area * MAX_BALL_AREA_RATIO
    debug = DetectionDebug(active_colors=len(active_colors))

    for color in active_colors:
        debug.masks_checked += 1
        mask = make_mask(hsv, color)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            debug.contours_seen += 1
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            minimum_ball_area = minimum_ball_area_for_y(
                center_y,
                frame.shape[0],
                frame_area
            )

            if area <= minimum_ball_area:
                debug.rejected_small += 1
                continue

            if area >= maximum_ball_area:
                debug.rejected_large += 1
                continue

            if not is_spherical(contour, w, h, mask):
                debug.rejected_shape += 1
                continue

            if not is_known_or_auto_color(color) and not has_sampled_color_variety(contour, hsv, color):
                debug.rejected_samples += 1
                continue

            center_x = x + w // 2
            radius = int(max(w, h) / 2)
            confidence = clamp(area / float(frame_area * 0.12), 0.0, 1.0)

            cv2.circle(
                detected_ball_mask,
                (center_x, center_y),
                radius,
                255,
                -1
            )

            targets.append(
                VisionTarget(
                    label=color["name"],
                    center_x=center_x,
                    center_y=center_y,
                    area=area,
                    confidence=confidence,
                    box=(x, y, w, h),
                    radius=radius,
                    color=color["box_color"]
                )
            )
            debug.accepted += 1

    raw_target_count = len(targets)
    targets = cull_overlapping_targets(targets)
    debug.rejected_overlap = raw_target_count - len(targets)
    debug.accepted = len(targets)

    return targets, detected_ball_mask, debug


def score_target_priority(target, targets, frame_width, frame_height, frame_area):
    distance_score = clamp(target.center_y / float(frame_height), 0.0, 1.0)
    area_score = clamp(target.area / float(frame_area * CLOSE_BALL_AREA_RATIO), 0.0, 1.0)
    center_score = 1.0 - clamp(
        abs(target.center_x - frame_width / 2.0) / (frame_width / 2.0),
        0.0,
        1.0
    )
    cluster_radius = max(1.0, frame_width * TARGET_CLUSTER_RADIUS_RATIO)
    cluster_area = target.area
    neighbor_count = 0

    for other in targets:
        if other is target:
            continue

        distance = np.hypot(
            target.center_x - other.center_x,
            target.center_y - other.center_y
        )

        if distance <= cluster_radius:
            closeness = 1.0 - distance / cluster_radius
            cluster_area += other.area * closeness
            neighbor_count += 1

    cluster_area_score = clamp(
        cluster_area / float(frame_area * CLOSE_BALL_AREA_RATIO),
        0.0,
        1.0
    )
    cluster_count_score = clamp(neighbor_count / 3.0, 0.0, 1.0)
    cluster_score = clamp(
        cluster_area_score * 0.65 + cluster_count_score * 0.35,
        0.0,
        1.0
    )
    priority_score = (
        distance_score * TARGET_DISTANCE_WEIGHT
        + cluster_score * TARGET_CLUSTER_WEIGHT
        + area_score * TARGET_AREA_WEIGHT
        + center_score * TARGET_CENTER_WEIGHT
    )

    return {
        "score": priority_score,
        "distance": distance_score,
        "cluster": cluster_score,
        "area": area_score,
        "center": center_score,
        "neighbors": neighbor_count
    }


def choose_best_target(targets, frame_width, frame_height, debug=None):
    if len(targets) == 0:
        return None

    frame_area = frame_width * frame_height
    scored_targets = []

    for target in targets:
        priority = score_target_priority(
            target,
            targets,
            frame_width,
            frame_height,
            frame_area
        )
        scored_targets.append((priority, target))

    scored_targets = sorted(
        scored_targets,
        key=lambda item: (
            item[0]["score"],
            item[0]["distance"],
            item[0]["cluster"],
            item[1].area,
            -abs(item[1].center_x - frame_width / 2.0)
        ),
        reverse=True
    )
    best_priority, best_target = scored_targets[0]

    if debug is not None:
        debug.priority_score = best_priority["score"]
        debug.priority_distance = best_priority["distance"]
        debug.priority_cluster = best_priority["cluster"]
        debug.priority_area = best_priority["area"]
        debug.priority_center = best_priority["center"]
        debug.priority_neighbors = best_priority["neighbors"]

    return best_target


class TargetStabilizer:
    def __init__(self):
        self.target = None
        self.last_seen_time = 0
        self.acquired_time = 0
        self.switch_candidate = None
        self.switch_candidate_frames = 0

    def select_target(self, targets, frame_width, frame_height, current_time, debug):
        debug.raw_target_count = len(targets)

        if len(targets) == 0:
            return self.hold_or_clear(current_time, debug)

        frame_area = frame_width * frame_height
        scored_targets = self.score_targets(targets, frame_width, frame_height, frame_area)
        raw_best_priority, raw_best_target = scored_targets[0]
        locked_match = self.find_locked_match(targets, frame_width)

        if self.target is None:
            return self.lock_target(raw_best_target, raw_best_priority, current_time, debug)

        if locked_match is None:
            if current_time - self.last_seen_time <= TARGET_HOLD_SECONDS:
                if self.should_switch(raw_best_target, raw_best_priority, None, frame_width):
                    return self.lock_target(raw_best_target, raw_best_priority, current_time, debug)

                return self.hold_or_clear(current_time, debug)

            return self.lock_target(raw_best_target, raw_best_priority, current_time, debug)

        locked_priority = score_target_priority(
            locked_match,
            targets,
            frame_width,
            frame_height,
            frame_area
        )

        if raw_best_target is not locked_match and self.should_switch(
            raw_best_target,
            raw_best_priority,
            locked_priority,
            frame_width
        ):
            return self.lock_target(raw_best_target, raw_best_priority, current_time, debug)

        self.switch_candidate = None
        self.switch_candidate_frames = 0
        return self.lock_target(locked_match, locked_priority, current_time, debug)

    def score_targets(self, targets, frame_width, frame_height, frame_area):
        scored_targets = []

        for target in targets:
            priority = score_target_priority(
                target,
                targets,
                frame_width,
                frame_height,
                frame_area
            )
            scored_targets.append((priority, target))

        return sorted(
            scored_targets,
            key=lambda item: (
                item[0]["score"],
                item[0]["distance"],
                item[0]["cluster"],
                item[1].area,
                -abs(item[1].center_x - frame_width / 2.0)
            ),
            reverse=True
        )

    def find_locked_match(self, targets, frame_width):
        if self.target is None:
            return None

        lock_radius = max(24.0, frame_width * TARGET_LOCK_RADIUS_RATIO)
        closest_target = None
        closest_distance = None

        for target in targets:
            distance = np.hypot(
                target.center_x - self.target.center_x,
                target.center_y - self.target.center_y
            )

            if distance > lock_radius:
                continue

            if closest_distance is None or distance < closest_distance:
                closest_target = target
                closest_distance = distance

        return closest_target

    def should_switch(self, candidate, candidate_priority, current_priority, frame_width):
        if self.target is None:
            return True

        current_score = 0.0

        if current_priority is not None:
            current_score = current_priority["score"]

        if candidate_priority["score"] < current_score + TARGET_SWITCH_MARGIN:
            self.switch_candidate = None
            self.switch_candidate_frames = 0
            return False

        if self.switch_candidate is not None:
            distance = np.hypot(
                candidate.center_x - self.switch_candidate.center_x,
                candidate.center_y - self.switch_candidate.center_y
            )
        else:
            distance = None

        if distance is None or distance > max(24.0, frame_width * TARGET_LOCK_RADIUS_RATIO):
            self.switch_candidate = candidate
            self.switch_candidate_frames = 1
        else:
            self.switch_candidate = candidate
            self.switch_candidate_frames += 1

        return self.switch_candidate_frames >= TARGET_SWITCH_FRAMES

    def lock_target(self, target, priority, current_time, debug):
        if self.target is None:
            smoothed_target = target
            self.acquired_time = current_time
        else:
            smoothed_target = self.smooth_target(target)

        self.target = smoothed_target
        self.last_seen_time = current_time
        self.fill_debug(priority, current_time, False, debug)
        return self.target

    def smooth_target(self, target):
        alpha = clamp(TARGET_SMOOTHING, 0.0, 1.0)
        return VisionTarget(
            label=target.label,
            center_x=int(round(self.target.center_x * (1.0 - alpha) + target.center_x * alpha)),
            center_y=int(round(self.target.center_y * (1.0 - alpha) + target.center_y * alpha)),
            area=self.target.area * (1.0 - alpha) + target.area * alpha,
            confidence=self.target.confidence * (1.0 - alpha) + target.confidence * alpha,
            box=target.box,
            radius=int(round(self.target.radius * (1.0 - alpha) + target.radius * alpha)),
            color=target.color
        )

    def hold_or_clear(self, current_time, debug):
        if self.target is not None and current_time - self.last_seen_time <= TARGET_HOLD_SECONDS:
            self.fill_debug(None, current_time, True, debug)
            return self.target

        self.target = None
        self.switch_candidate = None
        self.switch_candidate_frames = 0
        debug.stable_target_locked = False
        debug.stable_target_held = False
        debug.stable_target_label = "none"
        debug.stable_target_age = 0.0
        debug.switch_candidate_frames = 0
        return None

    def fill_debug(self, priority, current_time, held, debug):
        debug.stable_target_locked = self.target is not None
        debug.stable_target_held = held
        debug.stable_target_label = self.target.label if self.target is not None else "none"
        debug.stable_target_age = current_time - self.acquired_time
        debug.switch_candidate_frames = self.switch_candidate_frames

        if priority is not None:
            debug.priority_score = priority["score"]
            debug.priority_distance = priority["distance"]
            debug.priority_cluster = priority["cluster"]
            debug.priority_area = priority["area"]
            debug.priority_center = priority["center"]
            debug.priority_neighbors = priority["neighbors"]


class DriveStabilizer:
    def __init__(self):
        self.last_steering = 0.0
        self.last_time = 0

    def smooth(self, command, current_time, debug):
        debug.raw_steering = command.steering

        if self.last_time == 0:
            self.last_time = current_time
            self.last_steering = command.steering
            debug.smoothed_steering = command.steering
            debug.steering_limited = False
            return command

        elapsed = max(0.001, current_time - self.last_time)
        max_change = STEERING_SLEW_RATE * elapsed
        steering_delta = command.steering - self.last_steering
        limited_delta = clamp(steering_delta, -max_change, max_change)
        smoothed_steering = self.last_steering + limited_delta

        self.last_time = current_time
        self.last_steering = smoothed_steering
        debug.smoothed_steering = smoothed_steering
        debug.steering_limited = abs(limited_delta - steering_delta) > 0.001

        return DriveCommand(
            smoothed_steering,
            command.throttle,
            command.mode,
            command.reason
        )


def make_drive_command(best_target, frame_width, frame_area, last_seen_time, current_time):
    if best_target is None:
        if current_time - last_seen_time <= LOST_TARGET_TIMEOUT:
            return DriveCommand(0.0, 0.0, "assist", "coasting after recent target")

        return DriveCommand(0.0, 0.0, "assist", "no target")

    normalized_error = (best_target.center_x - frame_width / 2.0) / (frame_width / 2.0)

    if abs(normalized_error) < STEERING_DEADBAND:
        steering = 0.0
    else:
        steering = clamp(normalized_error * STEERING_GAIN, -1.0, 1.0)

    area_ratio = best_target.area / float(frame_area)

    if area_ratio >= CLOSE_BALL_AREA_RATIO:
        throttle = 0.0
        reason = "target close"
    else:
        throttle = min(MAX_TRIAL_THROTTLE, THROTTLE_HARD_LIMIT)
        reason = "seeking " + best_target.label

    return DriveCommand(steering, throttle, "assist", reason)


def draw_runtime_overlay(frame, targets, best_target, command, debug):
    frame_center_x = frame.shape[1] // 2

    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, frame.shape[0]),
        (255, 255, 255),
        1
    )

    for target in targets:
        x, y, w, h = target.box
        thickness = 4 if target == best_target else 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            target.color,
            thickness
        )
        cv2.circle(
            frame,
            (target.center_x, target.center_y),
            5,
            target.color,
            -1
        )


def print_telemetry(best_target, command, debug):
    debug_text = (
        "colors="
        + str(debug.active_colors)
        + " contours="
        + str(debug.contours_seen)
        + " ok="
        + str(debug.accepted)
        + " small="
        + str(debug.rejected_small)
        + " large="
        + str(debug.rejected_large)
        + " shape="
        + str(debug.rejected_shape)
        + " sample="
        + str(debug.rejected_samples)
        + " overlap="
        + str(debug.rejected_overlap)
        + " auto_candidates="
        + str(debug.auto_candidates)
        + " auto_profiles="
        + str(debug.auto_profiles)
        + " priority="
        + str(round(debug.priority_score, 3))
        + " priority_distance="
        + str(round(debug.priority_distance, 3))
        + " priority_cluster="
        + str(round(debug.priority_cluster, 3))
        + " priority_neighbors="
        + str(debug.priority_neighbors)
        + " stable_locked="
        + str(debug.stable_target_locked)
        + " stable_held="
        + str(debug.stable_target_held)
        + " stable_label="
        + debug.stable_target_label
        + " switch_frames="
        + str(debug.switch_candidate_frames)
        + " raw_steering="
        + str(round(debug.raw_steering, 3))
        + " smoothed_steering="
        + str(round(debug.smoothed_steering, 3))
        + " steering_limited="
        + str(debug.steering_limited)
    )

    if best_target is None:
        print(
            "Telemetry:",
            "target=none",
            "steering=" + str(round(command.steering, 3)),
            "throttle=" + str(round(command.throttle, 3)),
            "reason=" + command.reason,
            debug_text
        )
        return

    print(
        "Telemetry:",
        "target=" + best_target.label,
        "x=" + str(best_target.center_x),
        "y=" + str(best_target.center_y),
        "area=" + str(int(best_target.area)),
        "confidence=" + str(round(best_target.confidence, 2)),
        "steering=" + str(round(command.steering, 3)),
        "throttle=" + str(round(command.throttle, 3)),
        "reason=" + command.reason,
        debug_text
    )


def draw_calibration_menu(frame):
    cv2.putText(
        frame,
        "3 clicks, A=add, D=delete, R=area, X=reset area.",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y = 60

    if len(colors) == 0:
        cv2.putText(
            frame,
            "No colors added yet.",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 25
    else:
        for color in colors:
            menu_text = "Detecting: " + color["name"]

            cv2.putText(
                frame,
                menu_text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color["box_color"],
                2
            )

            y += 25

    cv2.putText(
        frame,
        last_assignment,
        (10, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    sample_text = (
        "Samples: "
        + str(len(calibration_samples))
        + "/"
        + str(samples_needed)
    )

    cv2.putText(
        frame,
        sample_text,
        (10, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if naming_mode:
        cv2.putText(
            frame,
            "Name: " + typed_color_name,
            (10, y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    if delete_mode:
        cv2.putText(
            frame,
            "Delete: " + typed_delete_name,
            (10, y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


def draw_detection_area(frame):
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Detection Area",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    if roi_mode and roi_start is not None and roi_end is not None:
        cv2.rectangle(
            frame,
            roi_start,
            roi_end,
            (255, 255, 255),
            2
        )


target_stabilizer = TargetStabilizer()
drive_stabilizer = DriveStabilizer()


try:
    while True:
        ret, frame = camera.read()

        if not ret:
            dashboard.close()
            print_camera_read_failure(camera)
            break

        current_time = time.time()
        if last_frame_time > 0:
            frame_delta = current_time - last_frame_time

            if frame_delta > 0:
                current_fps = 1.0 / frame_delta

        last_frame_time = current_time
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        auto_colors = auto_calibrator.update(frame, hsv, current_time)
        active_colors = auto_colors + KNOWN_BALL_COLORS + colors
        targets, detected_ball_mask, debug = detect_tennis_balls(frame, hsv, active_colors)
        debug.auto_candidates = auto_calibrator.last_candidate_count
        debug.auto_profiles = len(auto_colors)
        best_target = target_stabilizer.select_target(
            targets,
            frame.shape[1],
            frame.shape[0],
            current_time,
            debug
        )

        if best_target is not None:
            last_target_seen_time = current_time

        raw_command = make_drive_command(
            best_target,
            frame.shape[1],
            frame.shape[0] * frame.shape[1],
            last_target_seen_time,
            current_time
        )
        command = drive_stabilizer.smooth(raw_command, current_time, debug)
        actuators.apply(command)

        dashboard.draw(
            frame,
            best_target,
            command,
            debug,
            auto_colors,
            current_time,
            current_fps
        )

        if not dashboard.enabled and current_time - last_detection_print_time >= TELEMETRY_INTERVAL:
            print_telemetry(best_target, command, debug)
            last_detection_print_time = current_time

        if not HEADLESS:
            draw_runtime_overlay(frame, targets, best_target, command, debug)
            cv2.imshow(window_name, frame)

            # Keeps OpenCV windows responsive; runtime key input is ignored.
            cv2.waitKey(1)

except KeyboardInterrupt:
    dashboard.close()
    print("Keyboard interrupt received. Neutralizing outputs and exiting.")
finally:
    dashboard.close()
    actuators.close()
    camera.release()

    if not HEADLESS:
        cv2.destroyAllWindows()
