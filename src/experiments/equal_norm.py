"""Magnitude-controlled latent interventions, not native semantic counterfactuals."""

import numpy as np


def future_delta(base, donor):
    base, donor = np.asarray(base), np.asarray(donor)
    if base.shape != donor.shape or base.ndim != 5 or base.shape[2] < 2:
        raise ValueError("Matching 5D video latents with future slices required")
    if not np.isfinite(base).all() or not np.isfinite(donor).all():
        raise ValueError("Nonfinite latent")
    return donor[:, :, 1:].astype(np.float64) - base[:, :, 1:].astype(np.float64)


def radial_future(base, direction, radius):
    """Keep the observed anchor and set future-only Euclidean distance."""
    base = np.asarray(base)
    direction = np.asarray(direction, dtype=np.float64)
    if base.ndim != 5 or direction.shape != base[:, :, 1:].shape:
        raise ValueError("Direction must cover future slices only")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(base).all() or not np.isfinite(direction).all():
        raise ValueError("Nonfinite latent or direction")
    if not np.isfinite(radius) or radius <= 0 or norm <= 0:
        raise ValueError("Positive finite radius and nonzero direction required")
    out = base.astype(np.float32).copy()
    out[:, :, 1:] = base[:, :, 1:].astype(np.float64) + direction * (radius / norm)
    return out


def equal_norm_plan(base, same, alternative, *, fractions=(0.5, 1.0), random_seeds=(101, 102)):
    """Fix every arm from donors, never from recipient action effects.

    Radius <= both donor distances avoids extrapolating either donor ray.
    Interpolation and Gaussian controls can still be off the native manifold.
    The caller must measure the realized distance after inference-dtype casting.
    """
    fractions = tuple(fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(
        not np.isfinite(x) or not 0 < x <= 1 for x in fractions
    ):
        raise ValueError("Unique fractions in (0, 1] required")
    if len(set(random_seeds)) != len(random_seeds) or any(type(x) is not int or x < 0 for x in random_seeds):
        raise ValueError("Unique nonnegative integer random seeds required")
    directions = {"same": future_delta(base, same), "alternative": future_delta(base, alternative)}
    norms = {k: float(np.linalg.norm(v)) for k, v in directions.items()}
    radius = min(norms.values())
    if radius <= 0:
        raise ValueError("Both donors must differ from baseline")
    for seed in random_seeds:
        directions[f"random_{seed}"] = np.random.default_rng(seed).standard_normal(directions["same"].shape)
    arms, metadata = {}, {}
    for index, fraction in enumerate(fractions):
        for label, direction in directions.items():
            name = f"radius_{index}_{label}"
            arms[name] = radial_future(base, direction, radius * fraction)
            metadata[name] = {"direction": label, "fraction": float(fraction),
                              "requested_radius": radius * fraction,
                              "native_future": False, "semantic_review": "not established"}
    return arms, {"native_delta_norms": norms, "common_radius": radius, "arms": metadata}


def realized_geometry(base, intended, used):
    requested, realized = future_delta(base, intended), future_delta(base, used)
    a, b = float(np.linalg.norm(requested)), float(np.linalg.norm(realized))
    if a <= 0 or b <= 0:
        raise ValueError("Nonzero perturbations required")
    return {"requested_norm": a, "realized_norm": b, "relative_norm_error": abs(a-b)/a,
            "direction_cosine": float(np.clip(np.vdot(requested, realized)/(a*b), -1, 1)),
            "anchor_exact": bool(np.array_equal(np.asarray(base)[:, :, :1], np.asarray(used)[:, :, :1]))}
