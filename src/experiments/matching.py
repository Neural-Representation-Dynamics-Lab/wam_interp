"""Outcome-free native donor matching. No simulator or model imports."""
import numpy as np


def latent_delta_norm(future, baseline):
    a, b = np.asarray(future), np.asarray(baseline)
    if a.shape != b.shape or a.ndim != 5 or a.shape[2] < 2:
        raise ValueError('Require matching five-dimensional video latents with future slices')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite latent')
    return float(np.linalg.norm((a[:, :, 1:] - b[:, :, 1:]).astype(np.float64)))


def motion_features(points):
    a = np.asarray(points, dtype=float)
    if a.shape != (9, 2) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a >= 224):
        raise ValueError('Matching requires nine visible, finite pixel landmarks, not interpolated points')
    lengths = np.linalg.norm(np.diff(a, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-8:
        raise ValueError('Stationary trajectory cannot support motion matching')
    return {'path_length': total, 'endpoint_displacement': float(np.linalg.norm(a[-1] - a[0])),
            'normalized_progress': np.r_[0., np.cumsum(lengths) / total].tolist()}


def compare_match(candidate, alternative, rules):
    """Inputs contain reviewed donor motion, never recipient policy outcomes."""
    discrepancies = {}
    for field, rule in [('latent_delta_norm', 'latent_delta_relative_tolerance'),
                        ('path_length', 'path_length_relative_tolerance'),
                        ('endpoint_displacement', 'endpoint_displacement_relative_tolerance')]:
        a, b = float(candidate[field]), float(alternative[field])
        if not np.isfinite([a, b]).all() or a < 0 or b <= 1e-8 or rules[rule] <= 0:
            raise ValueError('Invalid matching metric or tolerance')
        discrepancies[field] = abs(a - b) / b / rules[rule]
    a, b = (np.asarray(x['normalized_progress']) for x in (candidate, alternative))
    if a.shape != (9,) or b.shape != (9,) or not np.isfinite([a, b]).all():
        raise ValueError('Invalid progress profiles')
    discrepancies['progress'] = float(np.max(np.abs(a-b))) / rules['normalized_progress_max_absolute_difference']
    return {'eligible': all(v <= 1 for v in discrepancies.values()),
            'maximum_normalized_discrepancy': float(max(discrepancies.values())),
            'discrepancies': discrepancies}


def choose_match(candidates, alternative, rules, *, exclude=()):
    comparisons = {name: compare_match(value, alternative, rules)
                   for name, value in sorted(candidates.items()) if name not in exclude}
    eligible = [name for name, value in comparisons.items() if value['eligible']]
    selected = min(eligible, key=lambda name: (comparisons[name]['maximum_normalized_discrepancy'], name)) if eligible else None
    return {'selected': selected, 'comparisons': comparisons,
            'outcome': 'matched' if selected else 'no_eligible_native_donor'}
