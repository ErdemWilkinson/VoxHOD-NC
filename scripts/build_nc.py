"""Build VoxHOD-NC tiles from teacher-labelled @dronoloji frames (local only).

Train videos (split.json) give, per frame:
  - up to 4 positive tiles around teacher people, crop size chosen so the person
    is 4-48 px on the 160 px tile (the scale VoxHOD is built for),
  - 2 random negative tiles that touch no person,
  - 1 hard negative where VoxHOD v3 fired (>= 0.6) far from any teacher person.
Test videos give grid tiles (every 3rd frame) at 320 and 640 px crops; a tile is
dropped if a person on it is outside 3-64 px, so it is neither positive nor negative.

    python build_nc.py D:/datasets/raw/dronoloji D:/datasets/voxhod D:/datasets/voxhod_nc
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 160
SIZES = np.array([240, 320, 480, 640, 800, 1080])


def cut(image, boxes, x0, y0, size):
    tile = np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE), Image.BILINEAR))
    if not len(boxes):
        return tile, np.zeros((0, 4), np.float32)
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (c[:, 0] >= x0) & (c[:, 0] < x0 + size) & (c[:, 1] >= y0) & (c[:, 1] < y0 + size)
    return tile, (np.clip(boxes[inside] - [x0, y0, x0, y0], 0, size) * (TILE / size)).astype(np.float32)


def touches(boxes, x0, y0, size):
    return bool(len(boxes)) and bool(((boxes[:, 2] > x0) & (boxes[:, 0] < x0 + size) &
                                      (boxes[:, 3] > y0) & (boxes[:, 1] < y0 + size)).any())


def ok_scale(kept, lo, hi):
    side = np.maximum(kept[:, 2] - kept[:, 0], kept[:, 3] - kept[:, 1]) if len(kept) else np.zeros(0)
    return bool(((side >= lo) & (side <= hi)).all())


def save(path, tiles, boxes):
    counts = np.array([len(b) for b in boxes], np.int32)
    flat = np.concatenate(boxes) if counts.sum() else np.zeros((0, 4), np.float32)
    np.savez(path, images=np.stack(tiles).astype(np.uint8), boxes=flat.astype(np.float32), counts=counts)
    print(path.name, len(tiles), "tiles,", int(counts.sum()), "persons,", int((counts > 0).sum()), "positive tiles", flush=True)


def main():
    raw, heridal, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((raw / "split.json").read_text())
    teacher = json.loads((raw / "labels_teacher.json").read_text())
    pre = json.loads((raw / "labels_prelabel.json").read_text())
    rng = np.random.default_rng(0)

    tiles, tboxes = [], []
    for video in [v for v in split["train"] if v in teacher]:
        for frame, people in teacher[video].items():
            image = Image.open(raw / "frames" / video / frame).convert("RGB")
            W, H = image.size
            boxes = np.array([p[:4] for p in people], np.float32).reshape(-1, 4)
            for i in rng.permutation(len(boxes))[:4]:
                side = max(boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1])
                valid = SIZES[(side * TILE / SIZES >= 4) & (side * TILE / SIZES <= 48) & (SIZES <= H)]
                if not len(valid):
                    continue
                s = int(rng.choice(valid))
                cx, cy = boxes[i].reshape(2, 2).mean(0) + rng.uniform(-0.4, 0.4, 2) * s
                x0, y0 = int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s))
                tile, kept = cut(image, boxes, x0, y0, s)
                if ok_scale(kept, 2, 64):
                    tiles.append(tile); tboxes.append(kept)
            negatives = 0
            for _ in range(8):
                s = int(rng.choice(SIZES[SIZES <= H]))
                x0, y0 = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
                if not touches(boxes, x0, y0, s):
                    tiles.append(cut(image, boxes, x0, y0, s)[0]); tboxes.append(np.zeros((0, 4), np.float32))
                    negatives += 1
                    if negatives == 2:
                        break
            hard = [c for c in pre[video][frame] if c[3] >= 0.6 and
                    not (len(boxes) and (np.abs((boxes[:, :2] + boxes[:, 2:]) / 2 - c[:2]).max(1) < 60).any())]
            if hard:
                c = hard[int(rng.integers(len(hard)))]
                s = int(min(H, max(240, c[2] / 6 * TILE)))
                x0 = int(np.clip(c[0] - s / 2 + rng.uniform(-0.3, 0.3) * s, 0, W - s))
                y0 = int(np.clip(c[1] - s / 2 + rng.uniform(-0.3, 0.3) * s, 0, H - s))
                if not touches(boxes, x0, y0, s):
                    tiles.append(cut(image, boxes, x0, y0, s)[0]); tboxes.append(np.zeros((0, 4), np.float32))
        print(" ", video, len(tiles), flush=True)
    nc_tiles, nc_boxes = tiles, tboxes
    save(out / "dronoloji_train.npz", nc_tiles, nc_boxes)

    h = np.load(heridal / "train.npz")
    offsets = np.concatenate([[0], np.cumsum(h["counts"])])
    all_tiles = list(h["images"]) + nc_tiles
    all_boxes = [h["boxes"][offsets[i]:offsets[i + 1]] for i in range(len(h["counts"]))] + nc_boxes
    save(out / "train.npz", all_tiles, all_boxes)
    del all_tiles, h

    for size in (320, 640):
        tiles, tboxes = [], []
        for video in [v for v in split["test"] if v in teacher]:
            for frame, people in teacher[video].items():
                if (int(frame[:5]) - 1) % 3:
                    continue
                image = Image.open(raw / "frames" / video / frame).convert("RGB")
                W, H = image.size
                boxes = np.array([p[:4] for p in people], np.float32).reshape(-1, 4)
                xs = list(range(0, W - size, size)) + [W - size]
                ys = list(range(0, H - size, size)) + [H - size]
                for y0 in ys:
                    for x0 in xs:
                        tile, kept = cut(image, boxes, x0, y0, size)
                        partial = touches(boxes, x0, y0, size) and not len(kept)
                        if ok_scale(kept, 3, 64) and not partial:
                            tiles.append(tile); tboxes.append(kept)
        save(out / f"dronoloji_test{size}.npz", tiles, tboxes)

    for name in ("val", "test", "wisard"):
        if not (out / f"{name}.npz").exists():
            os.link(heridal / f"{name}.npz", out / f"{name}.npz")


if __name__ == "__main__":
    main()
