import curses
import os
import time


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
STEERING_STEP = float(os.environ.get("MANUAL_STEERING_STEP", "0.05"))
THROTTLE_STEP = float(os.environ.get("MANUAL_THROTTLE_STEP", "0.02"))


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


def normalized_to_steering_degrees(command):
    command = clamp(command, -1.0, 1.0)

    if command < 0:
        return STEERING_CENTER_DEGREES + (
            STEERING_CENTER_DEGREES - STEERING_LEFT_DEGREES
        ) * command

    return STEERING_CENTER_DEGREES + (
        STEERING_RIGHT_DEGREES - STEERING_CENTER_DEGREES
    ) * command


def normalized_to_steering_pulse(command):
    return steering_degrees_to_pulse_us(normalized_to_steering_degrees(command))


def normalized_to_throttle_pulse(command):
    return normalized_to_pulse(
        clamp(command, -1.0, 1.0),
        THROTTLE_REVERSE_US,
        THROTTLE_NEUTRAL_US,
        THROTTLE_FORWARD_US
    )


class Pca9685Output:
    def __init__(self):
        import board
        import busio
        from adafruit_pca9685 import PCA9685

        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50
        self.last_steering_us = None
        self.last_throttle_us = None

    def set_pulse_us(self, channel, pulse_us):
        duty_cycle = int(clamp(pulse_us * 50 * 65535 / 1000000, 0, 65535))
        self.pca.channels[channel].duty_cycle = duty_cycle

    def apply(self, steering, throttle):
        steering_us = normalized_to_steering_pulse(steering)
        throttle_us = normalized_to_throttle_pulse(throttle)
        self.last_steering_us = steering_us
        self.last_throttle_us = throttle_us
        self.set_pulse_us(STEERING_CHANNEL, steering_us)
        self.set_pulse_us(THROTTLE_CHANNEL, throttle_us)

    def neutralize(self):
        self.apply(0.0, 0.0)

    def close(self):
        self.neutralize()
        self.pca.deinit()


def steering_bar(steering, width):
    label = "L "
    suffix = " R"
    bar_width = max(11, width - len(label) - len(suffix) - 1)
    center_index = bar_width // 2
    marker_index = int(round((clamp(steering, -1.0, 1.0) + 1.0) * 0.5 * (bar_width - 1)))
    bar = ["-"] * bar_width
    bar[center_index] = "|"
    bar[marker_index] = "#"
    return label + "".join(bar) + suffix


def throttle_bar(throttle, width):
    label = "R "
    suffix = " F"
    bar_width = max(11, width - len(label) - len(suffix) - 1)
    center_index = bar_width // 2
    marker_index = int(round((clamp(throttle, -1.0, 1.0) + 1.0) * 0.5 * (bar_width - 1)))
    bar = ["-"] * bar_width
    bar[center_index] = "|"
    bar[marker_index] = "#"
    return label + "".join(bar) + suffix


def draw(screen, steering, throttle, output, message):
    height, width = screen.getmaxyx()
    steering_degrees = normalized_to_steering_degrees(steering)
    lines = [
        "Manual PWM Control",
        "==================",
        "",
        "[Channels]",
        "steering channel=" + str(STEERING_CHANNEL)
        + " throttle channel=" + str(THROTTLE_CHANNEL),
        "steering pulse_us=" + str(output.last_steering_us)
        + " throttle pulse_us=" + str(output.last_throttle_us),
        "",
        "[Steering]",
        "value=" + str(round(steering, 3))
        + " degrees=" + str(round(steering_degrees, 1))
        + " L/C/R=" + str(STEERING_LEFT_DEGREES)
        + "/" + str(STEERING_CENTER_DEGREES)
        + "/" + str(STEERING_RIGHT_DEGREES),
        steering_bar(steering, width),
        "",
        "[Throttle]",
        "value=" + str(round(throttle, 3))
        + " reverse/neutral/forward us="
        + str(THROTTLE_REVERSE_US)
        + "/" + str(THROTTLE_NEUTRAL_US)
        + "/" + str(THROTTLE_FORWARD_US),
        throttle_bar(throttle, width),
        "",
        "[Keys]",
        "A center steering    S full left    D full right",
        "J/I throttle +step/full forward    L/M throttle -step/full reverse",
        "K neutral throttle   W/E steering -/+ step",
        "Space all neutral    Q quit",
        "",
        "[Status]",
        message,
    ]
    screen.erase()

    for index, line in enumerate(lines[:height - 1]):
        screen.addstr(index, 0, line[:max(0, width - 1)])

    screen.refresh()


def run(screen, output):
    steering = 0.0
    throttle = 0.0
    message = "Holding neutral for ESC arm seconds: " + str(ESC_ARM_SECONDS)
    output.neutralize()
    draw(screen, steering, throttle, output, message)
    time.sleep(ESC_ARM_SECONDS)
    message = "Ready. Wheels should be off the ground."
    screen.nodelay(True)

    while True:
        draw(screen, steering, throttle, output, message)
        key = screen.getch()

        if key == -1:
            time.sleep(0.03)
            continue

        key = chr(key).lower() if 0 <= key <= 255 else ""

        if key == "q":
            message = "Quitting; neutralizing outputs."
            break

        if key == " ":
            steering = 0.0
            throttle = 0.0
            message = "All neutral."
        elif key == "a":
            steering = 0.0
            message = "Steering centered."
        elif key == "s":
            steering = -1.0
            message = "Steering full left."
        elif key == "d":
            steering = 1.0
            message = "Steering full right."
        elif key == "w":
            steering = clamp(steering - STEERING_STEP, -1.0, 1.0)
            message = "Steering stepped left."
        elif key == "e":
            steering = clamp(steering + STEERING_STEP, -1.0, 1.0)
            message = "Steering stepped right."
        elif key == "i":
            throttle = 1.0
            message = "Throttle full forward."
        elif key == "k":
            throttle = 0.0
            message = "Throttle neutral."
        elif key == "m":
            throttle = -1.0
            message = "Throttle full reverse."
        elif key == "j":
            throttle = clamp(throttle + THROTTLE_STEP, -1.0, 1.0)
            message = "Throttle stepped forward."
        elif key == "l":
            throttle = clamp(throttle - THROTTLE_STEP, -1.0, 1.0)
            message = "Throttle stepped reverse."
        else:
            message = "Unknown key: " + repr(key)

        output.apply(steering, throttle)

    output.neutralize()


def main():
    output = Pca9685Output()

    try:
        curses.wrapper(run, output)
    finally:
        output.close()


if __name__ == "__main__":
    main()
