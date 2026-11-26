"""Read-only reanalysis of a saved bank; does not import a model or simulator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from wam_interp.experiments.instruction_sensitivity import balanced_contrasts, future_distance, semantic_pair


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def prepare(protocol_path, output):
    protocol=read(protocol_path); source=Path(protocol['source'])
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(protocol_path,output/'protocol.json')
    # Verify against the original completed download, not an edited index alone.
    manifest=read(source/'download_manifest.json'); inputs={}; records=[]; reviews={}
    def checked(p):
        relative=p.relative_to(source).as_posix()
        if relative not in manifest or sha(p)!=manifest[relative]['sha256']:
            raise ValueError(f'Source integrity failed: {relative}')
        inputs[relative]=manifest[relative]
    checked(source/'protocol.json'); checked(source/'analysis.json')
    for state in protocol['states']:
        case=source/f'state_{state}'; bank=case/'bank'
        for p in [bank/'candidate_index.json',bank/'context.json',case/'review.json']:checked(p)
        existing=read(case/'review.json')['candidates']; index=read(bank/'candidate_index.json')
        expected={f'{c}_seed_{n}' for c in protocol['conditions'] for n in protocol['seeds']}
        if {r['id'] for r in index}!=expected or len(index)!=len(expected):raise ValueError('Unexpected bank cells')
        for c in index:
            key=c['id'].rsplit('_seed_',1)[0]; cond=protocol['conditions'][key]
            if c['generation_request']['prompt']!=cond['prompt'] or c['seed']!=c['generation_request']['seed']:
                raise ValueError('Prompt/seed provenance mismatch')
            p=bank/f"{c['id']}.npz"; checked(p)
            if sha(p)!=c['sha256']:raise ValueError('Candidate index hash mismatch')
            uid=f"state_{state}/{c['id']}"
            record=dict(id=uid,candidate=c['id'],state=state,seed=c['seed'],**cond)
            records.append(record)
            for frame in range(9):checked(bank/f"{c['id']}_frames/{frame:03}.png")
            if c['id'] in existing:
                old=existing[c['id']]
                reviews[uid]={k:old[k] for k in ('label','notes','all_nine_frames_reviewed')}
                reviews[uid].update(origin='previous_AI_review',source_review_sha256=sha(case/'review.json'))
            else:
                reviews[uid]=dict(label=None,notes='',all_nine_frames_reviewed=False,origin='new_AI_review')
    contrasts=balanced_contrasts(records)
    write(output/'records.json',records);write(output/'contrasts.json',contrasts);write(output/'review.json',reviews)
    write(output/'input_manifest.json',inputs)
    scripts=['scripts/e1_instruction_sensitivity.py','src/experiments/instruction_sensitivity.py',
             'tests/test_e1_instruction_sensitivity.py','configs/experiments/e1_instruction_sensitivity.json']
    for name in scripts:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    write(output/'analysis_freeze.json',dict(at=now(),secondary_analysis=True,prior_reviews_known=27,
        source=str(source),protocol_sha256=sha(output/'protocol.json'),input_manifest_sha256=sha(output/'input_manifest.json'),
        records_sha256=sha(output/'records.json'),contrasts_sha256=sha(output/'contrasts.json'),
        source_files={name:sha(name) for name in scripts}))
    sheets=output/'sheets';sheets.mkdir()
    for i,r in enumerate(records):
        sheet=Image.new('RGB',(1344,750),'white');draw=ImageDraw.Draw(sheet)
        for frame in range(9):
            x,y=(frame%3)*448,(frame//3)*250
            draw.text((x+4,y+4),f"video {i:02}; frame {frame}; external | wrist",fill='black')
            with Image.open(source/f"state_{r['state']}/bank/{r['candidate']}_frames/{frame:03}.png") as im:
                if im.size!=(448,224):raise ValueError('Unexpected camera montage')
                sheet.paste(im,(x,y+25))
        sheet.save(sheets/f'{i:02}.jpg',quality=97)
    write(output/'sheet_mapping.json',{f'{i:02}':r['id'] for i,r in enumerate(records)})
    page='<!doctype html><meta charset="utf-8"><title>Instruction sensitivity: all 48 futures</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{width:100%}section{margin-bottom:30px}</style><h1>All 48 native futures</h1><p>Exploratory AI review; prior labels were known. No policy execution. Prompts are omitted from cards, but this is not blinded validation.</p>'
    for i in range(len(records)):page+=f'<section><h2>Video {i:02}</h2><img loading="lazy" src="sheets/{i:02}.jpg"></section>'
    (output/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps({'prepared':len(records),'new_videos_to_review':sum(not r['all_nine_frames_reviewed'] for r in reviews.values())}))


def analyze(output, require_review=False):
    protocol=read(output/'protocol.json');source=Path(protocol['source']);freeze=read(output/'analysis_freeze.json')
    if sha(output/'protocol.json')!=freeze['protocol_sha256'] or sha(output/'input_manifest.json')!=freeze['input_manifest_sha256']:
        raise ValueError('Frozen protocol/manifest changed')
    if sha(output/'records.json')!=freeze['records_sha256'] or sha(output/'contrasts.json')!=freeze['contrasts_sha256']:
        raise ValueError('Frozen factorial records changed')
    for name,digest in freeze['source_files'].items():
        if sha(name)!=digest:raise ValueError(f'Analysis source changed: {name}')
    for name,entry in read(output/'input_manifest.json').items():
        if sha(source/name)!=entry['sha256']:raise ValueError(f'Source changed: {name}')
    records=read(output/'records.json');pairs=balanced_contrasts(records)
    if pairs!=read(output/'contrasts.json'):raise ValueError('Contrasts changed')
    arrays={};frames={};reviews=read(output/'review.json');byid={r['id']:r for r in records}
    if set(reviews)!=set(byid):raise ValueError('Review coverage differs')
    complete=all(r.get('all_nine_frames_reviewed') and r.get('label') and r.get('notes') for r in reviews.values())
    if require_review and not complete:raise ValueError('Semantic review incomplete')
    for r in records:
        bank=source/f"state_{r['state']}/bank"
        with np.load(bank/f"{r['candidate']}.npz",allow_pickle=False) as d:arrays[r['id']]=d['future'].copy()
        frames[r['id']]=np.stack([np.asarray(Image.open(bank/f"{r['candidate']}_frames/{i:03}.png").convert('RGB'),dtype=np.float32) for i in range(1,9)])
    results=[]
    for pair in pairs:
        a,b=pair['a'],pair['b'];x,y=frames[a],frames[b]
        metrics=future_distance(arrays[a],arrays[b]);diff=np.abs(x-y)
        metrics.update(external_rgb_mae=float(diff[:,:,:224].mean()),wrist_rgb_mae=float(diff[:,:,224:].mean()))
        if complete and pair['kind']=='target_word':
            metrics.update(semantic_pair({**byid[a],**reviews[a]},{**byid[b],**reviews[b]}))
        results.append({**pair,**metrics})
    states=[]
    for state in protocol['states']:
        summary={'state':state,'comparisons':{}}
        for kind in ('target_word','action_wording','seed'):
            rows=[r for r in results if r['state']==state and r['kind']==kind]
            summary['comparisons'][kind]={'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('latent_rmse','external_rgb_mae','wrist_rgb_mae')},
                'exactly_equal':sum(r['future_exactly_equal'] for r in rows)}
        if complete:
            subset=[{**r,**reviews[r['id']]} for r in records if r['state']==state]
            rows=[r for r in results if r['state']==state and r['kind']=='target_word']
            summary['semantics']={'labels_by_intended_target':{t:dict(Counter(r['label'] for r in subset if r['target']==t)) for t in ('cream','butter')},
                'paired_denominator':len(rows),'both_follow_intended_target':sum(r['both_follow_intended_target'] for r in rows),
                'different_clear_targets':sum(r['different_clear_targets'] for r in rows),'both_labels_clear':sum(r['both_labels_clear'] for r in rows)}
        states.append(summary)
    overall={kind:{k:float(np.mean([s['comparisons'][kind][k] for s in states])) for k in ('latent_rmse','external_rgb_mae','wrist_rgb_mae')} for kind in ('target_word','action_wording','seed')}
    result=dict(at=now(),status='complete' if complete else 'numerical_complete_review_pending',native_futures=len(records),independent_states=len(states),
        new_gpu_inferences=0,new_policy_branches=0,all_input_hashes_passed=True,observed_anchors_identical=True,
        semantic_review_complete=complete,human_review='pending',states=states,equal_state_means=overall,
        limitations=protocol['limits'],review_sha256=sha(output/'review.json'))
    write(output/'pair_metrics.json',results);write(output/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','analyze'])
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_instruction_sensitivity.json'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--require-review',action='store_true')
    args=ap.parse_args()
    if args.stage=='prepare':prepare(args.protocol,args.output)
    else:analyze(args.output,args.require_review)
