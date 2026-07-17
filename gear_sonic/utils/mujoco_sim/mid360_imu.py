"""Livox Mid-360 IMU sampling and noise helpers.

The physical Mid-360 reports gyro in rad/s and acceleration in ``g`` at
200 Hz.  MuJoCo's site accelerometer reports m/s^2, so the simulation converts
units before applying a sensor-local noise model.  Mount/orientation correction
is deliberately left to the simulated Livox driver, matching the hardware
boundary where livox_ros_driver2 rotates both lidar and IMU data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

STANDARD_GRAVITY_M_S2 = 9.80665


class PhaseLockedSampler:
    """Select samples at an average fixed rate from a faster simulation clock.

    The 500 Hz physics loop cannot represent every 5 ms boundary exactly.  A
    phase-locked deadline produces the expected alternating 4/6 ms cadence
    without the long-term rate loss caused by ``next = now + period``.
    """

    def __init__(self, frequency_hz: float):
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        self.period_s = 1.0 / float(frequency_hz)
        self.next_sample_s = self.period_s

    def due(self, simulation_time_s: float) -> bool:
        now = float(simulation_time_s)
        if now + 1e-12 < self.next_sample_s:
            return False
        while self.next_sample_s <= now + 1e-12:
            self.next_sample_s += self.period_s
        return True

    def reset(self) -> None:
        self.next_sample_s = self.period_s


@dataclass(frozen=True)
class Mid360NoiseConfig:
    """Mid-360 noise parameters in the units emitted by the Livox driver."""

    frequency_hz: float = 200.0
    gyro_noise_density_rad_s_sqrt_hz: float = math.radians(0.02)
    accel_noise_density_g_sqrt_hz: float = 30e-6
    gyro_bias_std_rad_s: float = math.radians(10.0) / 3600.0
    accel_bias_std_g: float = 100e-6
    enabled: bool = True
    seed: int | None = None


class Mid360NoiseModel:
    """Per-episode bias plus white measurement noise for gyro/accelerometer."""

    def __init__(self, config: Mid360NoiseConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        # Density -> one-sample RMS under an ideal Nyquist-band white-noise
        # assumption.  These remain explicit/configurable because hardware
        # characterization may later provide measured discrete-time values.
        bandwidth_hz = 0.5 * config.frequency_hz
        self.gyro_sample_std = config.gyro_noise_density_rad_s_sqrt_hz * math.sqrt(
            bandwidth_hz
        )
        self.accel_sample_std = config.accel_noise_density_g_sqrt_hz * math.sqrt(
            bandwidth_hz
        )
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        if not self.config.enabled:
            self.gyro_bias.fill(0.0)
            self.accel_bias.fill(0.0)
            return
        self.gyro_bias = self._rng.normal(0.0, self.config.gyro_bias_std_rad_s, 3)
        self.accel_bias = self._rng.normal(0.0, self.config.accel_bias_std_g, 3)

    def apply(
        self, gyro_rad_s: np.ndarray, accel_m_s2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        gyro = np.asarray(gyro_rad_s, dtype=np.float64).copy()
        accel_g = np.asarray(accel_m_s2, dtype=np.float64) / STANDARD_GRAVITY_M_S2
        if self.config.enabled:
            gyro += self.gyro_bias + self._rng.normal(0.0, self.gyro_sample_std, 3)
            accel_g += self.accel_bias + self._rng.normal(0.0, self.accel_sample_std, 3)
        return gyro, accel_g
