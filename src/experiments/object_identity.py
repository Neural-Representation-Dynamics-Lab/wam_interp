"""Visual-asset transactions and behavioural target-identity criteria."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def exchanged_visual_positions(positions, geometry_ids, object_centres):
    """Translate two native visual assets between XY slots, preserving their Z.

    Call only during rendering, never around a physics or forward-dynamics call.
    Meshes, UVs, materials, orientations and collision geometry are untouched.
    """
    ids = np.asarray(geometry_ids, dtype=int)
    centres = np.asarray(object_centres, dtype=float)
    if ids.shape != (2,) or len(set(ids.tolist())) != 2 or centres.shape != (2, 3):
        raise ValueError('Two distinct visual geometries and two object centres required')
    if not np.isfinite(centres).all() or np.any(ids < 0) or np.any(ids >= len(positions)):
        raise ValueError('Invalid geometry IDs or nonfinite centres')
    saved = positions[ids].copy()
    delta = centres[::-1] - centres
    delta[:, 2] = 0
    try:
        positions[ids] = saved + delta
        yield
    finally:
        positions[ids] = saved


def identity_reversal(endpoints, axis, *, minimum_shift_m=.08, maximum_control_shift_m=.02):
    """Operational noun-selective response, not proof of an internal identity code."""
    required = {(q, v) for q in ('soup', 'tomato', 'cream') for v in ('original', 'swapped')}
    if set(endpoints) != required:
        raise ValueError('Complete three-instruction by two-image factorial required')
    e = {k: np.asarray(v, dtype=float) for k, v in endpoints.items()}
    u = np.asarray(axis, dtype=float)
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError('Finite nonzero 3D axis required')
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in e.values()):
        raise ValueError('Finite 3D endpoints required')
    u /= np.linalg.norm(u)
    original = float((e['tomato', 'original'] - e['soup', 'original']) @ u)
    swapped = float((e['tomato', 'swapped'] - e['soup', 'swapped']) @ u)
    changes = {q: float((e[q, 'swapped'] - e[q, 'original']) @ u) for q in ('soup', 'tomato')}
    control = float(np.linalg.norm(e['cream', 'swapped'] - e['cream', 'original']))
    criteria = {'native_noun_separation': original >= minimum_shift_m,
                'swapped_noun_separation': swapped <= -minimum_shift_m,
                'soup_follows_identity': changes['soup'] >= minimum_shift_m,
                'tomato_follows_identity': changes['tomato'] <= -minimum_shift_m,
                'untouched_target_stable': control <= maximum_control_shift_m}
    return {'original_tomato_minus_soup_m': original, 'swapped_tomato_minus_soup_m': swapped,
            'soup_swap_shift_m': changes['soup'], 'tomato_swap_shift_m': changes['tomato'],
            'cream_endpoint_change_m': control, 'criteria': criteria, 'pass': all(criteria.values())}
