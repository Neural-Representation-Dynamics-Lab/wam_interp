"""Outcome-independent calibration and state-level summaries for video interventions."""
import numpy as np


def validate_protocol(protocol):
    calibration = protocol['calibration_states']
    test = protocol['test_states']
    if len(set(calibration)) != len(calibration) or len(set(test)) != len(test):
        raise ValueError('State IDs must be unique within each partition')
    if set(calibration) & set(test):
        raise ValueError('Calibration and test states must be disjoint')
    if len(set(t['task_id'] for t in protocol['tasks'])) != len(protocol['tasks']):
        raise ValueError('Task IDs must be unique')
    for task in protocol['tasks']:
        if len(task['targets']) != 2:
            raise ValueError('Exactly two reference targets required')
        if set(test) & set(protocol['previously_exposed'].get(str(task['task_id']), [])):
            raise ValueError('Previously exposed state in test partition')
    if protocol['human_review'] != 'pending':
        raise ValueError('Human review cannot be inferred from automated checks')


def rms_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Equal finite arrays required')
    return float(np.sqrt(np.mean((b-a)**2)))


def choose_appearance(rows, candidates):
    """Choose nearest log-ratio to unit latent magnitude; never accept action data.

    One row per calibration state and level, with the two appearance edits
    averaged within state. Tasks have equal numbers of calibration states.
    Candidate order resolves ties toward the weaker intervention.
    """
    allowed = {'task', 'state', 'appearance', 'motion_rms', 'appearance_rms', 'valid'}
    if any(set(r) != allowed for r in rows):
        raise ValueError('Calibration accepts only representation/validity fields')
    pairs = {(r['task'], r['state']) for r in rows}
    scores = []
    for name in candidates:
        selected = [r for r in rows if r['appearance'] == name]
        if len(selected) != len(pairs) or {(r['task'],r['state']) for r in selected} != pairs:
            raise ValueError('Every candidate must cover the same calibration states')
        if not all(r['valid'] and r['motion_rms']>0 and r['appearance_rms']>0
                   and np.isfinite(r['motion_rms']) and np.isfinite(r['appearance_rms']) for r in selected):
            scores.append({'appearance':name,'eligible':False});continue
        ratios = [r['appearance_rms']/r['motion_rms'] for r in selected]
        score = float(np.mean(np.abs(np.log(ratios))))
        scores.append({'appearance':name,'eligible':True,'score':score,'state_ratios':ratios,
                       'geometric_mean_ratio':float(np.exp(np.mean(np.log(ratios))))})
    eligible = [r for r in scores if r['eligible']]
    if not eligible:raise ValueError('No valid calibrated appearance candidate')
    winner = min(eligible, key=lambda r:r['score'])
    return {'selected':winner['appearance'],'scores':scores,
            'method':'Minimum mean absolute log appearance/motion latent RMS ratio; no action outcomes used.',
            'interpretation':'Scalar magnitude calibration does not match spatial support, salience, or semantics.'}


def state_bootstrap(rows, field, *, seed=20261121, repeats=10000):
    """Average seeds within state; resample whole states separately within task."""
    if not rows:raise ValueError('No paired observations')
    grouped = {}
    seen = set()
    for row in rows:
        key=(row['task'],row['state'],row['seed'])
        if key in seen:raise ValueError('Duplicate task/state/seed observation')
        seen.add(key)
        if not np.isfinite(row[field]):raise ValueError('Nonfinite outcome')
        grouped.setdefault((row['task'],row['state']),[]).append(float(row[field]))
    rng=np.random.default_rng(seed);tasks={};samples=[]
    for task in sorted({k[0] for k in grouped}):
        states={str(s):float(np.mean(v)) for (t,s),v in grouped.items() if t==task}
        values=np.asarray(list(states.values()))
        draws=values[rng.integers(len(values),size=(repeats,len(values)))].mean(axis=1)
        tasks[str(task)]={'states':states,'n_states':len(values),'mean':float(values.mean()),
                          'percentile_95_interval':np.quantile(draws,[.025,.975]).tolist()}
        samples.append(draws)
    return {'tasks':tasks,'equal_task_mean':float(np.mean([v['mean'] for v in tasks.values()])),
            'stratified_percentile_95_interval':np.quantile(np.mean(samples,axis=0),[.025,.975]).tolist(),
            'bootstrap_unit':'initial state, after averaging recipient seeds; stratified by task',
            'bootstrap_seed':seed,'repeats':repeats,
            'caution':'Descriptive interval for this bounded replication; not a power guarantee or semantic equivalence test.'}
