
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TILES = ROOT / "outputs" / "tiles"
OUT = ROOT / "outputs" / "submission_images"
OUT.mkdir(parents=True, exist_ok=True)

for image_dir in sorted(TILES.glob("image_*")):

    image_id = image_dir.name

    with (image_dir / "metadata.json").open(
        "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    height, width = map(
        int, metadata["source_shape"]
    )

    tiles = metadata["tiles"]

    # Reconstruct original pixel array
    full = np.zeros(
        (height, width),
        dtype=np.float32
    )

    coverage = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for tile_info in tiles:

        tile = np.load(
            image_dir / tile_info["filename"],
            allow_pickle=False
        )

        x = int(tile_info["x_start"])
        y = int(tile_info["y_start"])

        w = int(tile_info["valid_width"])
        h = int(tile_info["valid_height"])

        full[y:y+h, x:x+w] = tile[:h, :w]
        coverage[y:y+h, x:x+w] += 1

    if not np.all(coverage == 1):
        raise RuntimeError(
            f"{image_id}: Missing or overlapping pixels"
        )

    # Use percentile stretching for visual inspection
    finite = full[np.isfinite(full)]

    if finite.size == 0:
        raise RuntimeError(
            f"{image_id}: No finite pixel values"
        )

    low, high = np.percentile(
        finite,
        [1, 99.9]
    )

    high = max(high, low + 1)

    preview = np.clip(
        (full - low) / (high - low),
        0, 1
    )

    # Export 8-bit grayscale PNG
    output_path = OUT / f"{image_id}.png"

    plt.imsave(
        output_path,
        preview,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    print(
        f"{image_id}: {width}x{height} | "
        f"coverage OK | saved"
    )

print("\nFULL-SIZE IMAGE EXPORT COMPLETED")
print("Output directory:", OUT)
print("Original tiles unchanged.")
