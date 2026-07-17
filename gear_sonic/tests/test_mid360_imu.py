from __future__ import annotations

import numpy as np

from gear_sonic.utils.mujoco_sim.mid360_imu import (
    STANDARD_GRAVITY_M_S2,
    Mid360NoiseConfig,
    Mid360NoiseModel,
    PhaseLockedSampler,
)


def test_phase_locked_sampler_hits_200hz_from_500hz_without_rate_drift():
    sampler = PhaseLockedSampler(200.0)
    times = [tick * 0.002 for tick in range(1, 501) if sampler.due(tick * 0.002)]

    assert len(times) == 200
    assert set(np.round(np.diff(times), 3)) == {0.004, 0.006}
    assert times[-1] == 1.0


def test_noise_disabled_only_converts_acceleration_to_livox_g_units():
    model = Mid360NoiseModel(Mid360NoiseConfig(enabled=False))

    gyro, accel = model.apply(
        np.array([0.1, -0.2, 0.3]),
        np.array([0.0, 0.0, -STANDARD_GRAVITY_M_S2]),
    )

    np.testing.assert_allclose(gyro, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(accel, [0.0, 0.0, -1.0])


def test_seeded_noise_model_has_fixed_episode_bias_and_finite_measurements():
    config = Mid360NoiseConfig(enabled=True, seed=7)
    model = Mid360NoiseModel(config)
    bias_before = (model.gyro_bias.copy(), model.accel_bias.copy())

    gyro, accel = model.apply(np.zeros(3), np.zeros(3))

    assert np.all(np.isfinite(gyro))
    assert np.all(np.isfinite(accel))
    np.testing.assert_allclose(model.gyro_bias, bias_before[0])
    np.testing.assert_allclose(model.accel_bias, bias_before[1])

    model.reset()
    assert not np.array_equal(model.gyro_bias, bias_before[0])
    assert not np.array_equal(model.accel_bias, bias_before[1])
