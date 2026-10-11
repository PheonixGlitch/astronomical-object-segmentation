
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter, map_coordinates

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

TILES = ROOT / "outputs" / "tiles"

INPUT_CSV = (
    ROOT / "outputs" / "streak_proposals"
    / "streak_proposals.csv"
)

OUT = ROOT / "outputs" / "profile_validation"
OUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# SETTINGS
# ==========================================

FLANK_DISTANCE = 6
MIN_LENGTH = 5
SAMPLES_PER_PIXEL = 2

results = []

# ==========================================
# LOAD STREAK PROPOSALS
# ==========================================

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8",
    newline=""
) as file:
    proposals = list(csv.DictReader(file))

print("Proposals to inspect:", len(proposals))

# ==========================================
# PROCESS PROPOSALS
# ==========================================

for row in proposals:

    image_id = row["image"]
    tile_name = row["tile"]
    proposal_id = int(row["proposal_id"])

    tile_path = TILES / image_id / tile_name

    image = np.load(tile_path).astype(np.float32)

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    dx = x2 - x1
    dy = y2 - y1

    length = float(np.hypot(dx, dy))

    if length < MIN_LENGTH:
        continue

    # Background subtraction
    background = median_filter(
        image,
        size=31,
        mode="reflect"
    )

    residual = image - background

    # Unit vector perpendicular to line
    nx = -dy / length
    ny = dx / length

    count = max(
        int(length * SAMPLES_PER_PIXEL),
        10
    )

    t = np.linspace(0, 1, count)

    xs = x1 + t * dx
    ys = y1 + t * dy

    height, width = image.shape

    def sample(offset):

        sample_x = xs + offset * nx
        sample_y = ys + offset * ny

        inside = (
            (sample_x >= 0)
            & (sample_x <= width - 1)
            & (sample_y >= 0)
            & (sample_y <= height - 1)
        )

        coordinates = np.vstack([
            np.clip(sample_y, 0, height - 1),
            np.clip(sample_x, 0, width - 1)
        ])

        values = map_coordinates(
            residual,
            coordinates,
            order=1,
            mode="nearest"
        )

        values[~inside] = np.nan
        return values

    center = sample(0)
    left = sample(-FLANK_DISTANCE)
    right = sample(FLANK_DISTANCE)

    flank = (left + right) / 2.0

    contrast = center - flank

    finite = np.isfinite(contrast)
    valid_contrast = contrast[finite]

    if valid_contrast.size == 0:
        continue

    # Exploratory descriptors:
    # not calibrated astronomical SNR.
    mean_contrast = float(
        np.mean(valid_contrast)
    )

    median_contrast = float(
        np.median(valid_contrast)
    )

    positive_fraction = float(
        np.mean(valid_contrast > 0)
    )

    # Longest consecutive run with
    # positive centerline contrast.
    longest_run = 0
    current_run = 0

    for value in contrast:
        if np.isfinite(value) and value > 0:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run
            )
        else:
            current_run = 0

    run_fraction = longest_run / count

    print(
        f"{image_id} #{proposal_id} | "
        f"Length={length:.1f} | "
        f"Mean contrast={mean_contrast:.2f} | "
        f"Positive={positive_fraction:.2f} | "
        f"Longest run={run_fraction:.2f}"
    )

    results.append({
        "image": image_id,
        "tile": tile_name,
        "proposal_id": proposal_id,
        "length": round(length, 2),
        "mean_contrast": round(mean_contrast, 3),
        "median_contrast": round(
            median_contrast, 3
        ),
        "positive_fraction": round(
            positive_fraction, 3
        ),
        "longest_positive_run_fraction": round(
            run_fraction, 3
        ),
        "status": "REVIEW_REQUIRED"
    })

    # ======================================
    # SAVE PROFILE VISUALIZATION
    # ======================================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4)
    )

    # Local image around the proposal
    padding = 20

    xmin = max(
        0, int(min(x1, x2)) - padding
    )
    xmax = min(
        width, int(max(x1, x2)) + padding + 1
    )
    ymin = max(
        0, int(min(y1, y2)) - padding
    )
    ymax = min(
        height, int(max(y1, y2)) + padding + 1
    )

    patch = image[ymin:ymax, xmin:xmax]

    low, high = np.percentile(
        patch, [1, 99.5]
    )

    axes[0].imshow(
        patch,
        cmap="gray",
        vmin=low,
        vmax=max(high, low + 1),
        extent=[xmin, xmax, ymax, ymin]
    )

    axes[0].plot(
        [x1, x2],
        [y1, y2],
        color="red",
        linewidth=1.5
    )

    axes[0].set_title(
        "Candidate centerline"
    )

    # Profile chart
    distances = t * length

    axes[1].plot(
        distances,
        center,
        label="Centerline"
    )

    axes[1].plot(
        distances,
        flank,
        label="Flank background"
    )

    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="gray"
    )

    axes[1].set_xlabel(
        "Distance along line (pixels)"
    )
    axes[1].set_ylabel(
        "Background-subtracted intensity"
    )
    axes[1].set_title(
        "Intensity profile"
    )
    axes[1].legend()

    fig.suptitle(
        f"{image_id} | Proposal {proposal_id}"
    )

    plt.tight_layout()

    preview_path = (
        OUT /
        f"{image_id}_proposal_{proposal_id:03d}.png"
    )

    plt.savefig(
        preview_path,
        dpi=140
    )
    plt.close(fig)

# ==========================================
# SAVE VALIDATION REPORT
# ==========================================

report = OUT / "streak_profile_report.csv"

columns = [
    "image",
    "tile",
    "proposal_id",
    "length",
    "mean_contrast",
    "median_contrast",
    "positive_fraction",
    "longest_positive_run_fraction",
    "status"
]

with open(
    report,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(results)

print("\nProfile validation completed.")
print("Profiles analyzed:", len(results))
print("Report saved:", report)
print("No YOLO annotations modified.")
