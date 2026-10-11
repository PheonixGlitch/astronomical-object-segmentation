
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    median_filter,
    gaussian_filter,
    map_coordinates
)

from astropy.stats import sigma_clipped_stats
from skimage.transform import probabilistic_hough_line


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

TILES_DIR = ROOT / "outputs" / "tiles"
OUTPUT_DIR = ROOT / "outputs" / "streak_proposals"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILES = [
    ("image_00", "tile_r00_c00.npy"),
    ("image_04", "tile_r00_c00.npy"),
    ("image_08", "tile_r00_c00.npy")
]

IMAGE_SIZE = 1024

# Hough detection settings
HOUGH_THRESHOLD = 10
HOUGH_LINE_LENGTH = 12
MIN_LINE_LENGTH = 15
LINE_GAP = 5
DETECTION_SIGMA = 2.5

# Intensity validation
MIN_CONTRAST_SCORE = 0.6
MIN_COHERENCE = 0.65
FLANK_DISTANCE = 6

# Geometric merging
MAX_ANGLE_DIFFERENCE = 15
MAX_PERP_DISTANCE = 5
MAX_ALONG_GAP = 12
MIN_GROUP_LENGTH = 15

# New support-based filtering
MIN_SUPPORTING_SEGMENTS = 3
MIN_SUPPORTED_LENGTH = 25

supported_records = []
weak_records = []


# ==========================================
# 2. GEOMETRY UTILITIES
# ==========================================

def line_geometry(line):

    x1, y1, x2, y2 = [
        float(value) for value in line[:4]
    ]

    start = np.array([x1, y1], dtype=float)
    end = np.array([x2, y2], dtype=float)

    vector = end - start
    length = float(np.linalg.norm(vector))

    if length <= 0:
        return None

    direction = vector / length

    angle = float(
        np.degrees(
            np.arctan2(direction[1], direction[0])
        ) % 180
    )

    return start, end, length, direction, angle


def angle_difference(angle_a, angle_b):

    difference = abs(angle_a - angle_b) % 180

    return min(difference, 180 - difference)


def can_merge(line_a, line_b):

    geometry_a = line_geometry(line_a)
    geometry_b = line_geometry(line_b)

    if geometry_a is None or geometry_b is None:
        return False

    a0, a1, _, direction, angle_a = geometry_a
    b0, b1, _, _, angle_b = geometry_b

    if angle_difference(
        angle_a,
        angle_b
    ) > MAX_ANGLE_DIFFERENCE:
        return False

    normal = np.array([
        -direction[1],
        direction[0]
    ])

    perpendicular_distance = max(
        abs(float(np.dot(b0 - a0, normal))),
        abs(float(np.dot(b1 - a0, normal)))
    )

    if perpendicular_distance > MAX_PERP_DISTANCE:
        return False

    projection_a = sorted([
        float(np.dot(a0, direction)),
        float(np.dot(a1, direction))
    ])

    projection_b = sorted([
        float(np.dot(b0, direction)),
        float(np.dot(b1, direction))
    ])

    gap = max(
        0.0,
        projection_b[0] - projection_a[1],
        projection_a[0] - projection_b[1]
    )

    return gap <= MAX_ALONG_GAP


# ==========================================
# 3. INTENSITY VALIDATION
# ==========================================

def validate_line(line, residual, noise):

    x1, y1, x2, y2, length = line

    if length <= 0:
        return False, 0.0, 0.0

    dx = x2 - x1
    dy = y2 - y1

    nx = -dy / length
    ny = dx / length

    sample_count = max(
        int(length * 2),
        10
    )

    t = np.linspace(
        0.0,
        1.0,
        sample_count
    )

    xs = x1 + t * dx
    ys = y1 + t * dy

    def sample(offset):

        coordinates = np.vstack([
            ys + offset * ny,
            xs + offset * nx
        ])

        return map_coordinates(
            residual,
            coordinates,
            order=1,
            mode="nearest"
        )

    center = sample(0)
    left = sample(-FLANK_DISTANCE)
    right = sample(FLANK_DISTANCE)

    flank = (left + right) / 2
    contrast = center - flank

    score = (
        float(np.mean(contrast))
        / max(float(noise), 1e-6)
    )

    coherence = float(
        np.mean(contrast > 0)
    )

    accepted = (
        score >= MIN_CONTRAST_SCORE
        and coherence >= MIN_COHERENCE
    )

    return accepted, score, coherence


# ==========================================
# 4. GROUP RELATED HOUGH SEGMENTS
# ==========================================

def group_segments(lines):

    count = len(lines)
    parent = list(range(count))

    def find(index):

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(first, second):

        root_a = find(first)
        root_b = find(second)

        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(count):

        for j in range(i + 1, count):

            if can_merge(lines[i], lines[j]):
                union(i, j)

    groups = {}

    for index, line in enumerate(lines):

        root = find(index)

        groups.setdefault(
            root,
            []
        ).append(line)

    return list(groups.values())


# ==========================================
# 5. PCA CENTERLINE FITTING
# ==========================================

def fit_group(group):

    points = []

    for x1, y1, x2, y2, *_ in group:

        points.append([x1, y1])
        points.append([x2, y2])

    points = np.asarray(
        points,
        dtype=np.float64
    )

    center = points.mean(axis=0)
    centered = points - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False
    )

    direction = vh[0]

    projections = centered @ direction

    start = (
        center
        + projections.min() * direction
    )

    end = (
        center
        + projections.max() * direction
    )

    length = float(
        np.linalg.norm(end - start)
    )

    return start, end, length


# ==========================================
# 6. CREATE PROPOSAL RECORD
# ==========================================

def proposal_record(
    image_id,
    filename,
    proposal_id,
    proposal,
    status
):

    start, end, length, segment_count = proposal

    midpoint = (start + end) / 2

    return {
        "image": image_id,
        "tile": filename,
        "proposal_id": proposal_id,
        "x1": round(float(start[0]), 2),
        "y1": round(float(start[1]), 2),
        "x2": round(float(end[0]), 2),
        "y2": round(float(end[1]), 2),
        "length": round(length, 2),
        "merged_segments": segment_count,
        "center_x": round(float(midpoint[0]), 2),
        "center_y": round(float(midpoint[1]), 2),
        "status": status
    }


# ==========================================
# 7. PROCESS THE THREE TEST TILES
# ==========================================

for image_id, filename in TILES:

    print(f"\nProcessing {image_id}/{filename}")

    tile_path = (
        TILES_DIR / image_id / filename
    )

    metadata_path = (
        TILES_DIR / image_id / "metadata.json"
    )

    if not tile_path.exists():
        raise FileNotFoundError(tile_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    image = np.load(
        tile_path
    ).astype(np.float32)

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    tile_info = next(
        (
            tile for tile in metadata["tiles"]
            if tile["filename"] == filename
        ),
        None
    )

    if tile_info is None:
        raise ValueError(
            f"Tile metadata missing: {filename}"
        )

    valid_height = int(
        tile_info["valid_height"]
    )

    valid_width = int(
        tile_info["valid_width"]
    )

    if not (
        1 <= valid_height <= IMAGE_SIZE
        and 1 <= valid_width <= IMAGE_SIZE
    ):
        raise ValueError(
            "Invalid tile dimensions."
        )

    valid_mask = np.zeros_like(
        image,
        dtype=bool
    )

    valid_mask[
        :valid_height,
        :valid_width
    ] = True

    # --------------------------------------
    # BACKGROUND SUBTRACTION
    # --------------------------------------

    background = median_filter(
        image,
        size=31,
        mode="reflect"
    )

    residual = image - background

    smoothed = gaussian_filter(
        residual,
        sigma=1.0
    )

    _, smooth_median, smooth_std = sigma_clipped_stats(
        smoothed[valid_mask],
        sigma=3.0,
        maxiters=5
    )

    smooth_noise = max(
        float(smooth_std),
        1e-6
    )

    threshold = (
        float(smooth_median)
        + DETECTION_SIGMA * smooth_noise
    )

    binary = (
        (smoothed > threshold)
        & valid_mask
    )

    # --------------------------------------
    # HOUGH LINE DETECTION
    # --------------------------------------

    raw_lines = probabilistic_hough_line(
        binary,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_LINE_LENGTH,
        line_gap=LINE_GAP,
        rng=42
    )

    # --------------------------------------
    # ORIGINAL RESIDUAL NOISE
    # --------------------------------------

    _, _, residual_std = sigma_clipped_stats(
        residual[valid_mask],
        sigma=3.0,
        maxiters=5
    )

    residual_noise = max(
        float(residual_std),
        1e-6
    )

    # --------------------------------------
    # INTENSITY-VALIDATED SEGMENTS
    # --------------------------------------

    validated_lines = []

    for start, end in raw_lines:

        x1, y1 = start
        x2, y2 = end

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1
            )
        )

        if length < MIN_LINE_LENGTH:
            continue

        line = (
            x1, y1,
            x2, y2,
            length
        )

        accepted, score, coherence = validate_line(
            line,
            residual,
            residual_noise
        )

        if accepted:
            validated_lines.append(
                (*line, score, coherence)
            )

    # --------------------------------------
    # MERGE VALIDATED SEGMENTS
    # --------------------------------------

    groups = group_segments(validated_lines)

    supported_proposals = []
    weak_proposals = []

    for group in groups:

        start, end, length = fit_group(group)

        if length < MIN_GROUP_LENGTH:
            continue

        proposal = (
            start,
            end,
            length,
            len(group)
        )

        # Stronger experimental screening.
        if (
            len(group) >= MIN_SUPPORTING_SEGMENTS
            and length >= MIN_SUPPORTED_LENGTH
        ):
            supported_proposals.append(proposal)

        else:
            weak_proposals.append(proposal)

    # --------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------

    print("Raw Hough segments:", len(raw_lines))

    print(
        "Validated segments:",
        len(validated_lines)
    )

    print(
        "Supported streak proposals:",
        len(supported_proposals)
    )

    print(
        "Weak proposals retained for review:",
        len(weak_proposals)
    )

    # --------------------------------------
    # SAVE PROPOSAL RECORDS
    # --------------------------------------

    for index, proposal in enumerate(
        supported_proposals
    ):

        supported_records.append(
            proposal_record(
                image_id,
                filename,
                index,
                proposal,
                "REVIEW_REQUIRED"
            )
        )

    for index, proposal in enumerate(
        weak_proposals
    ):

        weak_records.append(
            proposal_record(
                image_id,
                filename,
                index,
                proposal,
                "WEAK_REVIEW"
            )
        )

    # --------------------------------------
    # VISUALIZE SUPPORTED PROPOSALS
    # --------------------------------------

    low, high = np.percentile(
        image[valid_mask],
        [1, 99.9]
    )

    high = max(
        float(high),
        float(low) + 1
    )

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest"
    )

    for start, end, length, count in supported_proposals:

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="red",
            linewidth=1.8
        )

    ax.set_title(
        f"{image_id} | "
        f"Supported streak proposals: "
        f"{len(supported_proposals)}"
    )

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)

    plt.tight_layout()

    preview_path = (
        OUTPUT_DIR
        / f"{image_id}_streak_proposals.png"
    )

    plt.savefig(
        preview_path,
        dpi=150
    )

    plt.close(fig)

    print("Preview saved:", preview_path)


# ==========================================
# 8. SAVE CSV REPORTS
# ==========================================

CSV_COLUMNS = [
    "image",
    "tile",
    "proposal_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "length",
    "merged_segments",
    "center_x",
    "center_y",
    "status"
]


def save_csv(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()
        writer.writerows(rows)


supported_csv = (
    OUTPUT_DIR / "streak_proposals.csv"
)

weak_csv = (
    OUTPUT_DIR / "weak_streak_proposals.csv"
)

save_csv(
    supported_csv,
    supported_records
)

save_csv(
    weak_csv,
    weak_records
)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n====================================")
print("GENERAL STREAK DETECTION COMPLETED")
print("====================================")

print(
    "Total supported proposals:",
    len(supported_records)
)

print(
    "Total weak proposals:",
    len(weak_records)
)

print("Supported CSV:", supported_csv)
print("Weak review CSV:", weak_csv)

print(
    "All streak proposals require manual review."
)

print(
    "Existing YOLO annotations were not modified."
)
