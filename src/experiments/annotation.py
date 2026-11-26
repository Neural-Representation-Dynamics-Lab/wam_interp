"""Offline manual landmark annotation and explicit, paired coordinate analysis."""
import json
from pathlib import Path

import numpy as np

from .trajectories import trajectory_metrics


def make_annotation_page(pairs_path):
    pairs_path = Path(pairs_path)
    spec = json.loads(pairs_path.read_text())
    payload = json.dumps(spec).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Trajectory annotation</title>
<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 16px;background:#f5f5f0;color:#202820}
canvas{outline:1px solid #888;width:100%;height:auto;cursor:crosshair}button,select{margin:8px;padding:8px}
pre{white-space:pre-wrap} .row{display:flex;gap:25px}.row>div{flex:1;min-width:0}</style>
<h1>Imagined and executed trajectories</h1>
<p>Click the same anatomical gripper landmark or object landmark in both images.
Only the fixed external camera (left half of the montage) is shown.
Coordinates are normalized to this view. Mark occluded points as missing.
Keep landmark definitions fixed across conditions. This is manual measurement, not ground truth.</p>
<select id="landmark"><option>gripper</option><option>object</option></select>
<button id="prev">Previous frame</button><button id="next">Next frame</button><span id="position"></span>
<div class="row"><div><p>Imagined</p><canvas id="imagined"></canvas><br><button id="mi">Missing imagined</button></div>
<div><p>Executed</p><canvas id="executed"></canvas><br><button id="me">Missing executed</button></div></div>
<button id="draft">Save draft</button><label>Resume JSON <input id="load" type="file" accept=".json"></label>
<button id="save">Download completed annotations</button><pre id="status"></pre>
<script>
const spec = PAYLOAD;
let index=0; let records=spec.pairs.map(p=>({step:p.step,gripper:{},object:{}}));
const images={imagined:new Image(),executed:new Image()};
const landmark=document.getElementById('landmark');
function draw(side){const canvas=document.getElementById(side),img=images[side];
canvas.width=img.naturalWidth/2;canvas.height=img.naturalHeight;
const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,canvas.width,canvas.height,0,0,canvas.width,canvas.height);canvas.dataset.ready='true';
const point=records[index][landmark.value][side];if(point){ctx.fillStyle='#ff3030';ctx.beginPath();ctx.arc(point[0]*canvas.width,point[1]*canvas.height,4,0,7);ctx.fill();}}
function show(){document.getElementById('position').textContent=`Frame ${index+1}/${records.length}, control step ${records[index].step}`;
for(const side of ['imagined','executed']){document.getElementById(side).dataset.ready='false';images[side].onload=()=>draw(side);images[side].src=spec.pairs[index][side];}
document.getElementById('status').textContent=JSON.stringify(records[index],null,2);}
for(const side of ['imagined','executed'])document.getElementById(side).onclick=e=>{if(e.target.dataset.ready!=='true')return;const r=e.target.getBoundingClientRect();
records[index][landmark.value][side]=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];show();};
document.getElementById('mi').onclick=()=>{records[index][landmark.value].imagined=null;show();};
document.getElementById('me').onclick=()=>{records[index][landmark.value].executed=null;show();};
document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);show();};
document.getElementById('next').onclick=()=>{index=Math.min(records.length-1,index+1);show();};landmark.onchange=show;
function download(complete){if(complete&&records.some(r=>['gripper','object'].some(k=>!('imagined'in r[k])||!('executed'in r[k])))){alert('Annotate or mark missing both landmarks in every frame.');return;}
const data={...spec,records};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
a.download=spec.condition+(complete?'_annotations.json':'_draft.json');a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.getElementById('save').onclick=()=>download(true);document.getElementById('draft').onclick=()=>download(false);
document.getElementById('load').onchange=async e=>{try{const data=JSON.parse(await e.target.files[0].text());
if(data.state_id!==spec.state_id||data.condition!==spec.condition||data.context_id!==spec.context_id||JSON.stringify(data.pairs)!==JSON.stringify(spec.pairs)||data.records.length!==records.length)throw Error('Draft belongs to different frames or run');
if(data.records.some((r,i)=>r.step!==spec.pairs[i].step||['gripper','object'].some(k=>!r[k]||Object.values(r[k]).some(v=>v!==null&&(!Array.isArray(v)||v.length!==2||v.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))))))throw Error('Invalid coordinates');
records=data.records;show();}catch(err){alert(err.message);}};show();
</script>""".replace("PAYLOAD", payload)
    target = pairs_path.with_suffix(".html")
    target.write_text(html, encoding="utf-8")
    return target


def analyze_annotations(path):
    data = json.loads(Path(path).read_text())
    records = data["records"]
    if len(records) != len(data["pairs"]) or not records:
        raise ValueError("Annotation count differs from paired frames")
    times = [r["step"] for r in records]
    if times != [r["step"] for r in data["pairs"]]:
        raise ValueError("Annotation timestamps differ from frame pairs")
    result = {"synthetic": data["synthetic"], "state_id": data["state_id"],
              "context_id": data.get("context_id"),
              "condition": data["condition"], "coordinate_frame": data["coordinate_frame"],
              "units": data["units"], "time_units": "control_steps", "metrics": {}}
    for landmark in ("gripper", "object"):
        tracks = {}
        for side in ("imagined", "executed"):
            points = []
            for record in records:
                p = record[landmark][side]  # Missing key means incomplete annotation.
                if p is None:
                    points.append([np.nan, np.nan])
                elif len(p) == 2 and all(np.isfinite(p)) and all(0 <= v <= 1 for v in p):
                    points.append(p)
                else:
                    raise ValueError("Image coordinates must be normalized finite 2D points or null")
            tracks[side] = points
        result["metrics"][landmark] = trajectory_metrics(times, tracks["imagined"], tracks["executed"])
    return result
