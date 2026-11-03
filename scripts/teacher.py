"""Pseudo-label @dronoloji frames with a large COCO person detector (teacher).

torchvision Faster R-CNN ResNet50-FPN v2 (BSD-3 code, COCO weights), on GPU.
Each 1920x1080 frame is cut into a 3x2 grid of overlapping 720 px crops so small
people are seen near full resolution, plus the whole frame for large people.
Person boxes above SCORE are merged with NMS. Output labels_teacher.json
{video: {frame: [[x1, y1, x2, y2, score], ...]}}. Local data only.

    python teacher.py D:/datasets/raw/dronoloji
"""
import json
import os
import time
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.io import decode_jpeg, read_file
from torchvision.ops import nms

SCORE, CROP = 0.5, 720
STEP = int(os.environ.get("TEACHER_STEP", "3"))  # every STEP-th frame (frames are 2 s apart)
REST = float(os.environ.get("TEACHER_REST", "0.3"))  # seconds of GPU rest per frame, for temperature


def crops(width, height):
    xs = [0, (width - CROP) // 2, width - CROP]
    ys = [0, height - CROP]
    return [(x, y, CROP) for y in ys for x in xs]


@torch.no_grad()
def main():
    root = Path(sys.argv[1])
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE)
    model = model.eval().cuda()
    out_path = root / "labels_teacher.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    only = set(os.environ.get("TEACHER_VIDEOS", "").split()) or None
    for video in sorted(p for p in (root / "frames").iterdir() if p.is_dir()):
        if only and video.name not in only:
            continue
        frames = sorted(video.glob("*.jpg"))[::STEP]
        if video.name in out and len(out[video.name]) == len(frames):
            continue
        out[video.name] = {}
        for frame in frames:
            time.sleep(REST)
            image = decode_jpeg(read_file(str(frame)), device="cuda").float() / 255
            _, height, width = image.shape
            parts = [(0, 0, image)] + [(x, y, image[:, y:y + s, x:x + s]) for x, y, s in crops(width, height)]
            with torch.autocast("cuda", dtype=torch.float16):
                results = model([p[2] for p in parts])
            boxes, scores = [], []
            for (x0, y0, _), r in zip(parts, results):
                keep = r["labels"] == 1
                boxes.append(r["boxes"][keep].float() + torch.tensor([x0, y0, x0, y0], device="cuda"))
                scores.append(r["scores"][keep].float())
            boxes, scores = torch.cat(boxes), torch.cat(scores)
            keep = nms(boxes, scores, 0.5)
            out[video.name][frame.name] = [[round(v, 1) for v in b.tolist()] + [round(float(s), 3)]
                                           for b, s in zip(boxes[keep], scores[keep])]
        people = sum(len(v) for v in out[video.name].values())
        hits = sum(1 for v in out[video.name].values() if v)
        print(video.name, len(out[video.name]), "frames,", hits, "with people,", people, "boxes", flush=True)
        out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
