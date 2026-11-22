"""Export every existing contact chunk in both native views for a visibility audit.

This changes no simulator state or inference data. It does not estimate points
behind occlusion. Selection and point annotation must be separate review steps.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def export(source, output):
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for episode in sorted(source.glob('contact_state_*')):
        state = int(episode.name.split('_')[-1])
        status = json.loads((episode/'status.json').read_text())
        if status['status'] != 'complete': raise ValueError('Only completed episodes may be screened')
        for folder in sorted(episode.glob('chunk_*')):
            spec = json.loads((folder/'frame_pairs.json').read_text())
            with np.load(folder/'execution.npz', allow_pickle=False) as arrays:
                obj = arrays['object_alphabet_soup_1_main']
                relative = np.linalg.norm(obj-obj[0], axis=1)
            for camera, offset in [('agentview', 0), ('wrist', 224)]:
                sheet = Image.new('RGB', (1344, 750), 'white'); draw = ImageDraw.Draw(sheet)
                for i, pair in enumerate(spec['pairs']):
                    x, y = (i % 3)*448, (i//3)*250
                    draw.text((x+3, y+3), f"s{state} {folder.name} {camera} t{pair['step']} imagined | actual", fill='black')
                    for j, side in enumerate(('imagined', 'executed')):
                        im = Image.open(folder/pair[side]); assert im.size == (448, 224)
                        sheet.paste(im.crop((offset, 0, offset+224, 224)), (x+j*224, y+25))
                name = f'state_{state}_{folder.name}_{camera}.jpg'; sheet.save(output/name, quality=96)
                index.append({'state': state, 'chunk': int(folder.name.split('_')[-1]), 'camera': camera,
                    'sheet': name, 'source_pairs': str(folder/'frame_pairs.json'),
                    'sampled_controls': [p['step'] for p in spec['pairs']],
                    'sampled_soup_motion_m': [float(relative[p['step']]) for p in spec['pairs']],
                    'max_soup_motion_m': float(relative.max()), 'review': 'pending'})
    (output/'index.json').write_text(json.dumps(index, indent=2))
    with zipfile.ZipFile(output/'visibility_sheets.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.glob('*.jpg')): archive.write(p, p.name)
        archive.write(output/'index.json', 'index.json')
    print(json.dumps({'sheets': len(index), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); export(args.source, args.output)
