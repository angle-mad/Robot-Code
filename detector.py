import os
import platform
import time
import traceback

import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "auto").lower()
VALID_CAMERA_BACKENDS = ("auto", "picamera2", "opencv")
HEADLESS_REQUEST = os.environ.get("HEADLESS", "auto").lower()
VALID_HEADLESS_OPTIONS = ("auto", "true", "false")


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
    print("  frame size:", str(FRAME_WIDTH) + "x" + str(FRAME_HEIGHT))
    print("  python:", platform.python_version())
    print("  platform:", platform.platform())
    print("  opencv:", cv2.__version__)
    print("  /dev video devices:", ", ".join(list_video_devices()) or "none")
    print()


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
        return True, frame[:, :, :3]

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


camera = open_camera(FRAME_WIDTH, FRAME_HEIGHT)
window_name = "Tennis Ball Detector"
last_hsv_sample = None
last_click = None
last_assignment = "No color assigned yet"
current_hsv = None
last_detection_print_time = 0
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
    cv2.setMouseCallback(window_name, show_hsv_value)
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


def is_spherical(contour, width, height, mask):
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
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

    if circularity < 0.35 and circle_fill_ratio < 0.45:
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


while True:

    # Get image from webcam
    ret, frame = camera.read()

    if not ret:
        print_camera_read_failure(camera)
        print_camera_read_failure(camera)
        break


    # Convert camera image to HSV
    current_hsv = None
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )
    current_hsv = hsv


    detected_ball_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    current_time = time.time()

    for color in colors:

        mask = make_mask(hsv, color)


        # Find objects
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        for contour in contours:

            area = cv2.contourArea(contour)


            minimum_ball_area = frame.shape[0] * frame.shape[1] * 0.01

            # Ignore small objects
            if area > minimum_ball_area:

                x, y, w, h = cv2.boundingRect(contour)

                # Only highlight objects that look round like a ball.
                if not is_spherical(contour, w, h, mask):
                    continue

                # The object must contain multiple sampled shades from the ball.
                if not has_sampled_color_variety(contour, hsv, color):
                    continue

                # Draw box around the detected color
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    color["box_color"],
                    3
                )


                # Find center
                center_x = x + w//2
                center_y = y + h//2

                radius = int(max(w, h) / 2)

                cv2.circle(
                    detected_ball_mask,
                    (center_x, center_y),
                    radius,
                    255,
                    -1
                )


                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    color["box_color"],
                    -1
                )


                cv2.putText(
                    frame,
                    color["name"],
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color["box_color"],
                    2
                )

                if HEADLESS and current_time - last_detection_print_time > 1:
                    print(
                        "Detected",
                        color["name"],
                        "at x=" + str(center_x),
                        "y=" + str(center_y)
                    )
                    last_detection_print_time = current_time


    if not HEADLESS:
        draw_calibration_menu(frame)
        draw_detection_area(frame)


    if not HEADLESS and last_hsv_sample is not None and last_click is not None:

        cv2.circle(
            frame,
            last_click,
            8,
            (255, 255, 255),
            2
        )

        hsv_text = (
            "HSV: H="
            + str(last_hsv_sample[0])
            + " S="
            + str(last_hsv_sample[1])
            + " V="
            + str(last_hsv_sample[2])
        )

        cv2.putText(
            frame,
            hsv_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    if not HEADLESS:
        # Show camera
        cv2.imshow(
            window_name,
            frame
        )


        # Show mask
        cv2.imshow(
            "Detected Ball Mask",
            detected_ball_mask
        )


        key = cv2.waitKey(1) & 0xFF
    else:
        key = 255

    if key != 255:
        handle_key(key)


    # Press Q to quit
    if not naming_mode and not delete_mode and key == ord("q"):
        break


camera.release()

if not HEADLESS:
    cv2.destroyAllWindows()
