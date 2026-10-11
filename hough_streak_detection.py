
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, median_filter
from astropy.stats import sigma_clipped_stats
from skimage.transform import probabilistic_hough_line

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

TILE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "tiles"
    / "image_08"
    / "tile_r00_c00.npy"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "previews"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD IMAGE
# ==========================================

image = np.load(TILE_PATH).astype(np.float32)

if image.shape != (1024, 1024):
    raise ValueError(
        f"Unexpected image shape: {image.shape}"
    )

# ==========================================
# 3. BACKGROUND SUBTRACTION
# ==========================================

background = median_filter(
    image,
    size=31,
    mode="reflect"
)

residual = image - background

# Smooth only for line localization
smoothed = gaussian_filter(residual, sigma=1.0)

# ==========================================
# 4. ESTIMATE BACKGROUND NOISE
# ==========================================

mean, median, std = sigma_clipped_stats(
    smoothed,
    sigma=3.0,
    maxiters=5
)

noise_sigma = max(float(std), 1e-6)

# Exploratory threshold
threshold = float(median) + 2.5 * noise_sigma

binary = smoothed > threshold

print("Smoothed noise sigma:", round(noise_sigma, 3))
print("Candidate threshold:", round(threshold, 3))
print("Bright pixels:", int(binary.sum()))

# ==========================================
# 5. PROBABILISTIC HOUGH TRANSFORM
# ==========================================


lines = probabilistic_hough_line(
    binary,
    threshold=10,
    line_length=12,
    line_gap=5,
    rng=42
)


print("Hough line segments:", len(lines))

# ==========================================
# 6. FILTER SHORT LINE SEGMENTS
# ==========================================

MIN_LENGTH = 15

accepted_lines = []

for start, end in lines:

    x1, y1 = start
    x2, y2 = end

    length = np.hypot(
        x2 - x1,
        y2 - y1
    )

    if length >= MIN_LENGTH:
        accepted_lines.append(
            (start, end, float(length))
        )

print("Accepted line candidates:", len(accepted_lines))

# ==========================================
# 7. CHECK KNOWN STREAK REGION
# ==========================================

known_region_lines = []

for start, end, length in accepted_lines:

    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2

    if 20 <= mx <= 90 and 320 <= my <= 390:
        known_region_lines.append(
            (start, end, length)
        )

print(
    "Lines near known streak:",
    len(known_region_lines)
)

# ==========================================
# 8. VISUALIZATION
# ==========================================

low, high = np.percentile(image, [1, 99.9])

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax in axes:
    ax.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=max(high, low + 1)
    )
    ax.set_xlim(0, 1024)
    ax.set_ylim(1024, 0)

axes[0].set_title("Original FITS tile")

axes[1].set_title("Hough line candidates")

for start, end, length in accepted_lines:

    axes[1].plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="red",
        linewidth=1.5
    )

# Highlight the test region
from matplotlib.patches import Rectangle

axes[1].add_patch(
    Rectangle(
        (20, 320),
        70,
        70,
        fill=False,
        edgecolor="yellow",
        linewidth=1.5
    )
)

plt.tight_layout()

output_path = OUTPUT_DIR / "hough_streak_detection.png"

plt.savefig(output_path, dpi=180)
plt.close(fig)

print("Preview saved:", output_path)

# ==========================================
# 9. SAVE LINE COORDINATES
# ==========================================

line_data = np.array(
    [
        [
            start[0], start[1],
            end[0], end[1],
            length
        ]
        for start, end, length in accepted_lines
    ],
    dtype=np.float32
).reshape(-1, 5)

np.save(
    OUTPUT_DIR / "hough_line_candidates.npy",
    line_data
)

print("Line coordinates saved.")
