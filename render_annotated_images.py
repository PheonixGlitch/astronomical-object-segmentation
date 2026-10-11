"""Render color overlays as review PNGs from grayscale PNG + YOLO segmentation TXT.

Usage: python render_annotated_images.py PATH_TO_PLAIN_GRAYSCALE_PNGS PATH_TO_CORRECTED_LABELS PATH_TO_OUTPUT_IMAGES
Dependencies: Pillow, numpy. Colors: class 0 blue (36,112,255), class 1 red (255,42,70).
Review only; DO NOT use the annotated PNGs as segmentation model training images.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import sys

if len(sys.argv) != 4:
    raise SystemExit(__doc__)
source, labels, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)

# First 128 indices store an 8-bit gray backdrop quantized to even intensities.
colors = []
for i in range(128):
    g = 2 * i
    colors.extend((g, g, g))
colors.extend((36, 112, 255, 255, 42, 70))  # indices 128 = blue, 129 = red
colors.extend((0, 0, 0) * (256 - len(colors) // 3))

for image_path in sorted(source.glob('image_*.png')):
    label_path = labels / (image_path.stem + '.txt')
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    with Image.open(image_path) as im:
        base = np.array(im.convert('L'))
    height, width = base.shape
    rendered = Image.fromarray((base >> 1).copy(), mode='P')
    rendered.putpalette(colors)
    draw = ImageDraw.Draw(rendered)
    counts = [0, 0]
    for row, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        data = line.split()
        cls = int(data[0])
        if cls not in (0, 1):
            raise ValueError((label_path.name, row, 'Unexpected class'))
        nums = [float(a) for a in data[1:]]
        if len(nums) < 6 or len(nums) % 2:
            raise ValueError((label_path.name, row, 'Invalid polygon'))
        if not all(0 <= value <= 1 for value in nums):
            raise ValueError((label_path.name, row, 'Coordinate out of range'))
        points = [(round(nums[i] * width), round(nums[i + 1] * height))
                  for i in range(0, len(nums), 2)]
        draw.line(points + [points[0]], fill=128 if cls == 0 else 129,
                  width=2 if cls == 0 else 4, joint='curve')
        counts[cls] += 1
    outfile = output / image_path.name
    rendered.save(outfile, format='PNG', compress_level=1, optimize=False)
    print(outfile.name, 'blobs', counts[0], 'streaks', counts[1])
