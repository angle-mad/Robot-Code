# Autonomous RC-Car Trial Roadmap

## This Week's Trial Goals

1. Detect calibrated multi-colored tennis balls from the Pi camera.
2. Pick the best visible ball target.
3. Compute low-speed Ackermann steering and ESC throttle commands.
4. Keep actuator output disabled by default until hardware is ready.

## Electronics Assumptions

- Raspberry Pi 5 is the main computer.
- PCA9685 is connected over I2C.
- PCA9685 channel 0 drives the steering servo.
- PCA9685 channel 1 drives the RC ESC throttle input.
- The Pi, PCA9685, ESC/BEC, servo power, and motor battery must share ground.
- PCA9685 outputs are PWM control signals only. Do not power DC motors from the PCA9685.

## Runtime Modes

Dry-run vision and command telemetry:

```bash
python3 detector.py
```

Force headless telemetry:

```bash
HEADLESS=true python3 detector.py
```

Enable real PCA9685 actuator output only after bench testing:

```bash
ENABLE_ACTUATORS=true python3 detector.py
```

Useful tuning variables:

```bash
AUTO_CALIBRATE=true
AUTO_CALIBRATION_INTERVAL=1.0
AUTO_CALIBRATION_SATURATION_MIN=35
AUTO_CALIBRATION_VALUE_MIN=70
MIN_BALL_AREA_RATIO=0.004
KNOWN_COLOR_HUE_PADDING=12
KNOWN_COLOR_SATURATION_MIN=45
KNOWN_COLOR_VALUE_MIN=60
STEERING_CENTER_US=1500
STEERING_LEFT_US=1000
STEERING_RIGHT_US=2000
THROTTLE_NEUTRAL_US=1500
THROTTLE_FORWARD_US=1600
THROTTLE_REVERSE_US=1400
MAX_TRIAL_THROTTLE=0.18
STEERING_GAIN=1.25
STEERING_DEADBAND=0.06
CLOSE_BALL_AREA_RATIO=0.18
LOST_TARGET_TIMEOUT=0.5
```

## Safety Rules

- Test steering and ESC behavior with wheels off the ground first.
- Keep `ENABLE_ACTUATORS=false` until PWM ranges are confirmed.
- Use Ctrl-C to exit; the program neutralizes steering and throttle in its shutdown path.
- Add a physical kill switch before any fast or untethered run.

## Course Roadmap

- Ball acquisition: add a compliant roller intake after ball-seeking motion is reliable.
- Bucket scoring: detect the orange tape stripe, align to it, then run a timed shooter/feed sequence.
- Obstacle navigation: classify grey bricks, black PVC pipes, and wooden ramps into simple blocked/free regions.
- Ramp run: align to the ramp centerline, use steady throttle, and reduce steering gain while climbing.
- Final bar positioning: detect the bar, center under it, stop at a calibrated image position, then hand off to the climb mechanism.
