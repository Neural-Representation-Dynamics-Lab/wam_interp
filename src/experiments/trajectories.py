"""Timestamp-aligned trajectory metrics; missing landmarks remain missing."""

import numpy as np


def trajectory_metrics(times, imagined, executed, *, valid=None, executed_until=None):
    """Compare corresponding landmarks at identical, explicit timestamps.

    Arrays are [time, coordinate], in the same declared units/camera frame.
    Call per landmark. No interpolation, time warping, or camera registration
    is performed silently. NaNs mark missing detections on either side.
    """
    times = np.asarray(times, dtype=float)
    imagined, executed = (np.asarray(x, dtype=float) for x in (imagined, executed))
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Require nonempty finite timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    if imagined.shape != executed.shape or imagined.ndim != 2:
        raise ValueError("Require matching [time, coordinate] trajectories")
    if imagined.shape[0] != len(times) or imagined.shape[1] not in (2, 3):
        raise ValueError("Require one 2D or 3D coordinate per timestamp")
    eligible = np.ones(len(times), dtype=bool)
    if executed_until is not None:
        if not np.isfinite(executed_until) or executed_until < times[0]:
            raise ValueError("Executed prefix must include the first timestamp")
        eligible &= times <= executed_until
    mask = np.isfinite(imagined).all(1) & np.isfinite(executed).all(1) & eligible
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != times.shape:
            raise ValueError("Visibility mask must match timestamps")
        mask &= valid
    errors = np.linalg.norm(imagined - executed, axis=1)
    endpoint = np.flatnonzero(eligible)[-1]
    displacement_error = None
    stationary_ade = None
    if mask[0]:
        stationary_ade = float(np.linalg.norm(executed[mask] - executed[0], axis=1).mean())
    if mask[0] and mask[endpoint]:
        displacement_error = float(np.linalg.norm(
            (imagined[endpoint] - imagined[0]) - (executed[endpoint] - executed[0])))
    return {
        "eligible_frames": int(eligible.sum()), "valid_frames": int(mask.sum()),
        "coverage": float(mask.sum() / eligible.sum()),
        "ade": float(errors[mask].mean()) if mask.any() else None,
        "fde": float(errors[endpoint]) if mask[endpoint] else None,
        "displacement_error": displacement_error,
        "stationary_ade": stationary_ade,
        "times": times[eligible].tolist(),
        "errors": [float(errors[i]) if mask[i] else None for i in np.flatnonzero(eligible)],
    }
