
from pathlib import Path
import json
import math

import numpy as np
from astropy.io import fits

DATASET_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "tiles"

TILE_SIZE = 1024

fits_files = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
)

if len(fits_files) != 10:
    raise ValueError(f"Expected 10 FITS files, found {len(fits_files)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_tiles = 0

for image_index, image_path in enumerate(fits_files):

    print(f"\nProcessing image {image_index + 1}/10: {image_path.name}")

    # Load original unsigned pixel intensities
    with fits.open(image_path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.uint16)

    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image: {image_path.name}")

    height, width = image.shape

    columns = math.ceil(width / TILE_SIZE)
    rows = math.ceil(height / TILE_SIZE)

    image_output = OUTPUT_DIR / f"image_{image_index:02d}"
    image_output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_filename": image_path.name,
        "source_shape": [height, width],
        "tile_size": TILE_SIZE,
        "rows": rows,
        "columns": columns,
        "tiles": []
    }

    for row in range(rows):
        for col in range(columns):

            x0 = col * TILE_SIZE
            y0 = row * TILE_SIZE

            x1 = min(x0 + TILE_SIZE, width)
            y1 = min(y0 + TILE_SIZE, height)

            original_region = image[y0:y1, x0:x1]

            # Create an exact 1024 x 1024 tile
            tile = np.zeros(
                (TILE_SIZE, TILE_SIZE),
                dtype=np.uint16
            )

            valid_height = y1 - y0
            valid_width = x1 - x0

            tile[:valid_height, :valid_width] = original_region

            tile_name = f"tile_r{row:02d}_c{col:02d}.npy"
            tile_path = image_output / tile_name

            np.save(tile_path, tile)

            metadata["tiles"].append({
                "filename": tile_name,
                "row": row,
                "column": col,
                "x_start": x0,
                "y_start": y0,
                "valid_width": valid_width,
                "valid_height": valid_height
            })

            total_tiles += 1

    metadata_path = image_output / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Image size: {width} x {height}")
    print(f"Generated {rows * columns} tiles")

print(f"\nTotal tiles generated: {total_tiles}")
print(f"Saved to: {OUTPUT_DIR}")
