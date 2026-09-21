"""Render actual evaluated point clouds into a diagnostic GIF (not a UI recording)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
from webapp.bag_store import BagStore


def main():
    rows = [json.loads(line) for line in (ROOT / 'artifacts/evaluation/release-frames/doubleT_obstacle.jsonl').read_text().splitlines()]
    store = BagStore(ROOT / 'data/extracted/for_hackathon')
    bag = next(b for b in store.list_bags() if b['name'] == 'doubleT_obstacle')
    frames = []
    try:
        font = ImageFont.truetype('arial.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    for index in range(0, len(rows), 5):
        points = store.read_frame(bag['id'], index)['points']
        points = points[(abs(points[:,0]) < 4) & (points[:,1] >= 0) & (points[:,1] < 90) & (points[:,2] > -3) & (points[:,2] < 2)][::5]
        picture = Image.new('RGB', (1000, 650), '#091714')
        draw = ImageDraw.Draw(picture)
        draw.text((25,15), f'Metro Lidar Control | frame {index} | diagnostic replay', font=font, fill='#57e6a1')
        draw.text((25,45), 'Top view (left) / side view (right). Real bag + evaluated boxes.', font=font, fill='white')
        for x,y,z in points:
            draw.point((int(30+(x+4)/8*420),int(610-y/90*500)), fill='#65bab3')
            draw.point((int(520+y/90*450),int(610-(z+3)/5*500)), fill='#65bab3')
        for obj in rows[index]['objects']:
            low, high = obj['min_m'], obj['max_m']
            color = '#ff6272' if obj.get('track_confirmed') else '#ffbd59'
            draw.rectangle((30+(low['x']+4)/8*420,610-high['forward']/90*500,30+(high['x']+4)/8*420,610-low['forward']/90*500),outline=color,width=2)
            draw.rectangle((520+low['forward']/90*450,610-(high['z']+3)/5*500,520+high['forward']/90*450,610-(low['z']+3)/5*500),outline=color,width=2)
        frames.append(picture)
    target = ROOT / 'artifacts/release/diagnostic-replay.gif'
    target.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print(target)


if __name__ == '__main__':
    main()
