"""Outcome-free, bounded native-donor pair ranking and review gates."""
import numpy as np
from .matching import compare_match, motion_features


def rank_pairs(targets, norms, rules, limit, *, labels=('soup','cream')):
    if len(labels)!=2 or len(set(labels))!=2 or any(x in ('other','unclear') for x in labels):
        raise ValueError('Two distinct concrete target labels required')
    if type(limit) is not int or limit < 1 or not norms:
        raise ValueError('Positive review limit and recipient norms required')
    rows=[]
    for same in sorted(k for k,v in targets.items() if v==labels[0]):
        for alternative in sorted(k for k,v in targets.items() if v==labels[1]):
            differences={}
            for seed,values in norms.items():
                a,b=float(values[same]),float(values[alternative])
                if not np.isfinite([a,b]).all() or a<=0 or b<=0:
                    raise ValueError('Native donor norms must be positive and finite')
                differences[str(seed)]=abs(a-b)/b
            score=max(differences.values())/rules['latent_delta_relative_tolerance']
            rows.append({'same':same,'alternative':alternative,'norm_score':score,
                         'norm_eligible':score<=1,'relative_norm_differences':differences})
    eligible=sorted((r for r in rows if r['norm_eligible']),key=lambda r:(r['norm_score'],r['same'],r['alternative']))
    return {'all_pairs':rows,'shortlist':eligible[:limit],'norm_eligible_pairs':len(eligible),
            'unreviewed_norm_eligible_pairs':max(0,len(eligible)-limit)}


def select_pair(ranking, reviews, norms, rules, nuisance, *, labels=('soup','cream')):
    """Every shortlisted candidate must be reviewed, even if another pair fails."""
    candidates=set(n for p in ranking['shortlist'] for n in (p['same'],p['alternative']))
    for name in candidates:
        r=reviews[name]
        if r.get('all_nine_frames_reviewed') is not True or not r.get('notes'):
            raise ValueError(f'Incomplete native-video review: {name}')
        if r.get('label') not in (*labels,'other','unclear'):
            raise ValueError('Prompt intent cannot substitute for an explicit visual label')
        if r.get('acceptable') and r['label'] in labels:
            if r.get('landmarks_verified') is not True:
                raise ValueError(f'Accepted candidate needs visible, reviewed motion landmarks: {name}')
            motion_features(r['landmarks_pixels'])
    results=[]
    for pair in ranking['shortlist']:
        a,b=pair['same'],pair['alternative'];ra,rb=reviews[a],reviews[b]
        record={**pair,'key':f'{a}__{b}','eligible':False}
        if not (ra.get('acceptable') and rb.get('acceptable') and ra['label']==labels[0] and rb['label']==labels[1]):
            record['rejection']='Visual semantics/appearance did not support the intended contrast';results.append(record);continue
        checks={str(seed):compare_match({**motion_features(ra['landmarks_pixels']),'latent_delta_norm':values[a]},
            {**motion_features(rb['landmarks_pixels']),'latent_delta_norm':values[b]},rules) for seed,values in norms.items()}
        record['matching']=checks
        record['score']=max(c['maximum_normalized_discrepancy'] for c in checks.values())
        if not all(c['eligible'] for c in checks.values()):
            record['rejection']='Motion or latent matching failed';results.append(record);continue
        n=nuisance.get(record['key'],{})
        if n.get('review_complete') is not True or not n.get('notes'):
            raise ValueError('Numerically eligible pair still needs orientation/appearance review')
        record['nuisance']=n;record['eligible']=n.get('acceptable') is True
        if not record['eligible']:record['rejection']='Pair-level nuisance review failed'
        results.append(record)
    accepted=[r for r in results if r['eligible']]
    selected=min(accepted,key=lambda r:(r['score'],r['same'],r['alternative'])) if accepted else None
    return {'selected':selected,'reviewed_pairs':results,'outcome':'matched_native_pair' if selected else 'no_match_in_bounded_shortlist',
            'unreviewed_norm_eligible_pairs':ranking['unreviewed_norm_eligible_pairs']}


def target_direction(soup,cream):
    a,b=np.asarray(soup,dtype=float),np.asarray(cream,dtype=float)
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():
        raise ValueError('Finite world XYZ target positions required')
    vector=b-a;vector[2]=0
    norm=float(np.linalg.norm(vector))
    if norm<1e-8:raise ValueError('Targets have no distinct XY direction')
    return vector/norm
