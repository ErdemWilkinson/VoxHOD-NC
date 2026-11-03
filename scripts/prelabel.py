"""Run VoxHOD v3 (float) over the frames; build_nc.py uses its confident hits as hard negatives.

Each frame is cut into square grid crops at three sizes (320, 640, 1080 px),
resized to the 160 px model tile. Heatmap peaks above LOW become candidate
person centres in frame pixels; candidates from different scales closer than
MERGE px are merged (highest score kept). Output: labels_prelabel.json
{video: {frame: [[cx, cy, half, score], ...]}}. Local data only, never publish.

    python prelabel.py D:/datasets/raw/dronoloji path/to/voxhod.keras
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

TILE, STRIDE, LOW, MERGE = 160, 4, 0.25, 24
SCALES = (320, 640, 1080)


def grid(width, height, size):
    xs = list(range(0, width - size, size)) + [width - size]
    ys = list(range(0, height - size, size)) + [height - size]
    return [(x, y) for y in ys for x in xs]


def frame_tiles(image):
    width, height = image.size
    tiles, where = [], []
    for size in SCALES:
        size = min(size, width, height)
        for x0, y0 in grid(width, height, size):
            tiles.append(np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE), Image.BILINEAR)))
            where.append((x0, y0, size))
    return tiles, where


def candidates(heats, where):
    pooled = tf.nn.max_pool2d(heats, 3, 1, "SAME").numpy()
    found = []
    for heat, peak, (x0, y0, size) in zip(heats[..., 0], pooled[..., 0], where):
        ys, xs = np.nonzero((heat == peak) & (heat >= LOW))
        k = size / TILE
        for y, x in zip(ys, xs):
            found.append([x0 + (x + 0.5) * STRIDE * k, y0 + (y + 0.5) * STRIDE * k, 6 * k, float(heat[y, x])])
    found.sort(key=lambda c: -c[3])
    kept = []
    for c in found:
        if all((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 > MERGE ** 2 for d in kept):
            kept.append(c)
    return [[round(v, 1) for v in c[:3]] + [round(c[3], 3)] for c in kept]


def main():
    root, model = Path(sys.argv[1]), tf.keras.models.load_model(sys.argv[2], compile=False)
    out = {}
    for video in sorted(p for p in (root / "frames").iterdir() if p.is_dir()):
        out[video.name] = {}
        for frame in sorted(video.glob("*.jpg")):
            tiles, where = frame_tiles(Image.open(frame).convert("RGB"))
            heats = model.predict(np.stack(tiles).astype(np.float32) / 127.5 - 1.0, batch_size=64, verbose=0)
            out[video.name][frame.name] = candidates(heats, where)
        hits = sum(1 for v in out[video.name].values() if v)
        print(video.name, len(out[video.name]), "frames,", hits, "with candidates", flush=True)
        (root / "labels_prelabel.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
