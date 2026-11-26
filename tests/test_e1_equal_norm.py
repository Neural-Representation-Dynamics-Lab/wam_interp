import numpy as np
import pytest

from wam_interp.experiments.equal_norm import equal_norm_plan, future_delta, radial_future, realized_geometry


def fixtures():
    base = np.ones((1, 2, 3, 2, 2), dtype=np.float32)
    same, alt = base.copy(), base.copy()
    same[:, :, 1:] += 0.5
    alt[:, :, 1:] -= 2
    return base, same, alt


def test_radius_matches_without_extrapolation_and_preserves_anchor():
    base, same, alt = fixtures()
    arms, meta = equal_norm_plan(base, same, alt)
    assert len(arms) == 8
    for name, value in arms.items():
        assert np.array_equal(value[:, :, :1], base[:, :, :1])
        assert np.linalg.norm(future_delta(base, value)) == pytest.approx(meta['arms'][name]['requested_radius'])
    assert np.array_equal(arms['radius_1_same'], same)
    assert np.linalg.norm(future_delta(base, arms['radius_1_alternative'])) < np.linalg.norm(future_delta(base, alt))


def test_random_controls_are_reproducible_and_distinct():
    args = fixtures()
    first, _ = equal_norm_plan(*args)
    second, _ = equal_norm_plan(*args)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert not np.array_equal(first['radius_0_random_101'], first['radius_0_random_102'])


def test_realized_bfloat_geometry_audits_quantization():
    import torch
    base, same, alt = fixtures()
    arms, _ = equal_norm_plan(base, same, alt)
    intended = arms['radius_0_random_101']
    used = torch.tensor(intended).bfloat16().float().numpy()
    geometry = realized_geometry(base, intended, used)
    assert geometry['anchor_exact']
    assert geometry['direction_cosine'] > .99
    assert geometry['relative_norm_error'] < .02
    changed = used.copy(); changed[:, :, 0] += 1
    assert not realized_geometry(base, intended, changed)['anchor_exact']


@pytest.mark.parametrize('fractions', [[], [0], [1.1], [.5, .5], [float('nan')]])
def test_invalid_radii_rejected(fractions):
    with pytest.raises(ValueError): equal_norm_plan(*fixtures(), fractions=fractions)


def test_invalid_latents_and_zero_directions_rejected():
    base, same, alt = fixtures()
    with pytest.raises(ValueError): equal_norm_plan(base, base, alt)
    with pytest.raises(ValueError): radial_future(base, np.zeros_like(base[:, :, 1:]), 1)
    with pytest.raises(ValueError): future_delta(base, alt[:, :, :1])
    same[0, 0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError): equal_norm_plan(base, same, alt)


def test_historical_context_allows_versioned_source_but_not_asset_or_prompt_changes(monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('equal_norm_cli', root/'scripts/e1_equal_norm.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    prior = {'source_sha256': 'old', 'checkpoint': 'same', 'prompt': 'original', 'action_seed': 0}
    current = {**prior, 'source_sha256': 'new'}
    module.check_context(prior, current)
    with pytest.raises(ValueError, match='prompt'): module.check_context(prior, {**current, 'prompt': 'other'})
    with pytest.raises(ValueError, match='checkpoint'): module.check_context(prior, {**current, 'checkpoint': 'other'})
