
from pathlib import Path
import csv
import json
import re

import numpy as np
from PIL import Image

from scipy.ndimage import (
    median_filter,
    gaussian_filter,
    binary_dilation
)
from astropy.stats import sigma_clipped_stats
from skimage.measure import label, regionprops, find_contours
from skimage.draw import polygon2mask


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

TILES_DIR = ROOT / "outputs" / "tiles"
OUTPUT_DIR = ROOT / "outputs" / "full_batch_draft"

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in (IMAGE_DIR, LABEL_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

KNOWN_STREAK = (
    ROOT / "outputs" / "previews"
    / "merged_streak_mask.npy"
)

EXPECTED_IMAGES = 10
EXPECTED_TILES = 700
SIZE = 1024

# Experimental settings
BACKGROUND_WINDOW = 31
POINT_SIGMA = 1.0
POINT_THRESHOLD_SIGMA = 4.0
MIN_BLOB_AREA = 4

MIN_POLYGON_IOU = 0.95
STREAK_EXCLUSION_RADIUS = 3

# Class IDs are provisional.
CLASS_BLOB = 0
CLASS_STREAK = 1

# Candidate policies
EXPORT_BLOB_DRAFTS = True
INCLUDE_REVIEWED_STREAK = True

REVIEWED_STREAK_TILE = (
    "image_08",
    "tile_r00_c00.npy"
)

CSV_FIELDS = [
    "image",
    "tile",
    "instance",
    "class_id",
    "area",
    "polygon_iou",
    "exported",
    "review_reason"
]

summary_fields = [
    "image",
    "tile",
    "noise_sigma",
    "blob_candidates",
    "exported_blobs",
    "exported_streaks",
    "conflicts",
    "edge_flags",
    "rejected_polygons"
]


# ==========================================
# 2. LOAD VALID TILE DIMENSIONS
# ==========================================

def load_metadata(image_folder):

    path = image_folder / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if "tiles" not in metadata:
        raise ValueError(
            f"Missing tile metadata: {path}"
        )

    entries = {}

    for tile in metadata["tiles"]:

        filename = tile["filename"]

        if filename in entries:
            raise ValueError(
                f"Duplicate tile metadata: {filename}"
            )

        entries[filename] = tile

    return entries


def valid_dimensions(tile_info):

    height = int(tile_info["valid_height"])
    width = int(tile_info["valid_width"])

    if not (
        1 <= height <= SIZE
        and 1 <= width <= SIZE
    ):
        raise ValueError(
            f"Invalid tile dimensions: {height}, {width}"
        )

    return height, width


# ==========================================
# 3. ESTIMATE BACKGROUND
# ==========================================

def detect_blobs(image, valid_height, valid_width):

    # Process only the valid area, avoiding
    # padded pixels along image boundaries.
    valid_image = image[
        :valid_height,
        :valid_width
    ]

    background = median_filter(
        valid_image,
        size=BACKGROUND_WINDOW,
        mode="reflect"
    )

    residual = valid_image - background

    _, _, sigma = sigma_clipped_stats(
        residual,
        sigma=3.0,
        maxiters=5
    )

    noise = max(float(sigma), 1e-6)

    response = gaussian_filter(
        residual,
        sigma=POINT_SIGMA,
        mode="reflect"
    )

    binary = (
        response > POINT_THRESHOLD_SIGMA * noise
    )

    components = label(
        binary,
        connectivity=2
    )

    regions = []

    for region in regionprops(components):

        if region.area < MIN_BLOB_AREA:
            continue

        # Save a local bounding-box mask instead
        # of allocating a full 1024x1024 mask
        # for every individual source.
        min_y, min_x, max_y, max_x = region.bbox

        local_component = (
            components[min_y:max_y, min_x:max_x]
            == region.label
        )

        regions.append({
            "mask": local_component,
            "bbox": (
                min_y, min_x, max_y, max_x
            ),
            "area": int(region.area),
            "eccentricity": float(
                region.eccentricity
            )
        })

    return regions, noise


# ==========================================
# 4. POLYGON CONVERSION
# ==========================================

def mask_to_polygon(mask, origin_x=0, origin_y=0):

    padded = np.pad(
        mask.astype(np.uint8),
        1
    )

    contours = find_contours(
        padded,
        level=0.5
    )

    if len(contours) != 1:
        return None, 0.0

    contour = contours[0] - 1.0

    local_points = np.column_stack([
        np.clip(
            contour[:, 1],
            0, mask.shape[1] - 1
        ),
        np.clip(
            contour[:, 0],
            0, mask.shape[0] - 1
        )
    ])

    if len(local_points) > 1:

        keep = np.ones(
            len(local_points),
            dtype=bool
        )

        keep[1:] = np.any(
            np.diff(local_points, axis=0) != 0,
            axis=1
        )

        local_points = local_points[keep]

    if (
        len(local_points) > 1
        and np.allclose(
            local_points[0],
            local_points[-1]
        )
    ):
        local_points = local_points[:-1]

    if len(local_points) < 3:
        return None, 0.0

    reconstructed = polygon2mask(
        mask.shape,
        local_points[:, [1, 0]]
    )

    intersection = np.count_nonzero(
        mask & reconstructed
    )

    union = np.count_nonzero(
        mask | reconstructed
    )

    iou = (
        intersection / union
        if union else 0.0
    )

    points = local_points + np.array([
        origin_x,
        origin_y
    ])

    return points, float(iou)


def yolo_polygon(class_id, points):

    normalized = np.clip(
        points / SIZE,
        0.0,
        1.0
    )

    values = [str(class_id)]

    for x, y in normalized:

        values.extend([
            f"{x:.8f}",
            f"{y:.8f}"
        ])

    return " ".join(values)


# ==========================================
# 5. TILE BOUNDARY CHECK
# ==========================================

def touches_boundary(
    bbox,
    valid_height,
    valid_width
):

    min_y, min_x, max_y, max_x = bbox

    return (
        min_y == 0
        or min_x == 0
        or max_y >= valid_height
        or max_x >= valid_width
    )


# ==========================================
# 6. REVIEWED STREAK
# ==========================================

reviewed_streak_mask = None

if INCLUDE_REVIEWED_STREAK:

    if not KNOWN_STREAK.exists():
        raise FileNotFoundError(
            f"Reviewed streak missing: {KNOWN_STREAK}"
        )

    reviewed_streak_mask = np.load(
        KNOWN_STREAK,
        allow_pickle=False
    ).astype(bool)

    if reviewed_streak_mask.shape != (SIZE, SIZE):
        raise ValueError(
            "Reviewed streak mask has incorrect shape."
        )

    if not np.any(reviewed_streak_mask):
        raise ValueError(
            "Reviewed streak mask is empty."
        )


# ==========================================
# 7. PROCESS ONE TILE
# ==========================================

def process_tile(image_id, tile_path, metadata):

    tile_info = metadata[tile_path.name]

    valid_height, valid_width = valid_dimensions(
        tile_info
    )

    image = np.load(
        tile_path,
        allow_pickle=False
    ).astype(np.float32)

    if image.shape != (SIZE, SIZE):
        raise ValueError(
            f"Unexpected shape: {tile_path}"
        )

    valid = image[
        :valid_height,
        :valid_width
    ]

    if not np.all(np.isfinite(valid)):
        raise ValueError(
            f"Non-finite image pixels: {tile_path}"
        )

    stem = f"{image_id}_{tile_path.stem}"

    blobs, noise = detect_blobs(
        image,
        valid_height,
        valid_width
    )

    streaks = []

    if (
        INCLUDE_REVIEWED_STREAK
        and (image_id, tile_path.name)
        == REVIEWED_STREAK_TILE
    ):

        streaks.append(
            reviewed_streak_mask.copy()
        )

    # ======================================
    # STREAK EXCLUSION ZONE
    # ======================================

    exclusion = np.zeros(
        (SIZE, SIZE),
        dtype=bool
    )

    for streak in streaks:

        exclusion |= binary_dilation(
            streak,
            iterations=STREAK_EXCLUSION_RADIUS
        )

    annotations = []
    review_records = []

    conflicts = 0
    edge_flags = 0
    rejected_polygons = 0

    exported_blobs = 0
    exported_streaks = 0

    # ======================================
    # BLOB PROCESSING
    # ======================================

    for index, blob in enumerate(blobs):

        min_y, min_x, max_y, max_x = blob["bbox"]

        local_mask = blob["mask"]

        local_exclusion = exclusion[
            min_y:max_y,
            min_x:max_x
        ]

        conflict = bool(
            np.any(
                local_mask & local_exclusion
            )
        )

        if conflict:
            conflicts += 1

        edge = touches_boundary(
            blob["bbox"],
            valid_height,
            valid_width
        )

        if edge:
            edge_flags += 1

        points, iou = mask_to_polygon(
            local_mask,
            origin_x=min_x,
            origin_y=min_y
        )

        reasons = []

        if conflict:
            reasons.append("near_reviewed_streak")

        if edge:
            reasons.append("touches_tile_boundary")

        if blob["eccentricity"] >= 0.95:
            reasons.append("elongated_blob_review")

        if points is None:
            reasons.append("polygon_conversion_failed")

        elif iou < MIN_POLYGON_IOU:
            reasons.append("low_polygon_fidelity")

        exported = False

        # A blob close to a reviewed streak is
        # withheld instead of assigned blindly.
        if (
            EXPORT_BLOB_DRAFTS
            and not conflict
            and points is not None
            and iou >= MIN_POLYGON_IOU
        ):

            annotations.append(
                yolo_polygon(CLASS_BLOB, points)
            )

            exported_blobs += 1
            exported = True

        if points is None or iou < MIN_POLYGON_IOU:
            rejected_polygons += 1

        if not reasons:
            reasons = ["unreviewed_blob_candidate"]

        review_records.append({
            "image": image_id,
            "tile": tile_path.name,
            "instance": index,
            "class_id": CLASS_BLOB,
            "area": blob["area"],
            "polygon_iou": round(iou, 4),
            "exported": exported,
            "review_reason": ";".join(reasons)
        })

    # ======================================
    # REVIEWED STREAK PROCESSING
    # ======================================

    for index, streak in enumerate(streaks):

        mask = streak.copy()

        mask[valid_height:, :] = False
        mask[:, valid_width:] = False

        if not np.any(mask):
            continue

        ys, xs = np.nonzero(mask)

        min_y = int(ys.min())
        max_y = int(ys.max()) + 1
        min_x = int(xs.min())
        max_x = int(xs.max()) + 1

        local_mask = mask[
            min_y:max_y,
            min_x:max_x
        ]

        points, iou = mask_to_polygon(
            local_mask,
            origin_x=min_x,
            origin_y=min_y
        )

        if points is None or iou < MIN_POLYGON_IOU:
            raise ValueError(
                "Reviewed streak failed polygon validation."
            )

        annotations.append(
            yolo_polygon(
                CLASS_STREAK,
                points
            )
        )

        exported_streaks += 1

        review_records.append({
            "image": image_id,
            "tile": tile_path.name,
            "instance": index,
            "class_id": CLASS_STREAK,
            "area": int(mask.sum()),
            "polygon_iou": round(iou, 4),
            "exported": True,
            "review_reason": "reviewed_streak_candidate"
        })

    # ======================================
    # SAVE YOLO LABELS
    # ======================================

    label_path = LABEL_DIR / f"{stem}.txt"

    label_path.write_text(
        "\n".join(annotations)
        + ("\n" if annotations else ""),
        encoding="utf-8"
    )

    # ======================================
    # SAVE 8-BIT IMAGE
    # ======================================

    low, high = np.percentile(
        valid,
        [1, 99.9]
    )

    high = max(
        float(high),
        float(low) + 1.0
    )

    display = np.zeros(
        (SIZE, SIZE),
        dtype=np.uint8
    )

    scaled = np.clip(
        (valid - low) / (high - low),
        0,
        1
    )

    display[
        :valid_height,
        :valid_width
    ] = np.round(
        scaled * 255
    ).astype(np.uint8)

    Image.fromarray(
        display,
        mode="L"
    ).save(
        IMAGE_DIR / f"{stem}.png"
    )

    summary = {
        "image": image_id,
        "tile": tile_path.name,
        "noise_sigma": round(noise, 4),
        "blob_candidates": len(blobs),
        "exported_blobs": exported_blobs,
        "exported_streaks": exported_streaks,
        "conflicts": conflicts,
        "edge_flags": edge_flags,
        "rejected_polygons": rejected_polygons
    }

    return summary, review_records


# ==========================================
# 8. CSV REPORT HELPERS
# ==========================================

def write_csv(path, fields, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 9. FULL BATCH PROCESSING
# ==========================================

def main():

    image_folders = sorted(
        folder for folder in TILES_DIR.iterdir()
        if folder.is_dir()
        and re.fullmatch(
            r"image_\d+",
            folder.name
        )
    )

    if len(image_folders) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} image folders, "
            f"found {len(image_folders)}"
        )

    all_jobs = []

    for folder in image_folders:

        metadata = load_metadata(folder)

        actual = {
            path.name
            for path in folder.glob("tile_*.npy")
        }

        expected = set(metadata.keys())

        if actual != expected:
            raise ValueError(
                f"Tile inventory mismatch in {folder.name}"
            )

        for filename in sorted(expected):

            all_jobs.append((
                folder.name,
                folder / filename,
                metadata
            ))

    if len(all_jobs) != EXPECTED_TILES:
        raise ValueError(
            f"Expected {EXPECTED_TILES} tiles, "
            f"found {len(all_jobs)}"
        )

    # Prevent accidentally mixing old output
    # with a new run.
    existing = (
        list(LABEL_DIR.glob("*.txt"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if existing:
        raise RuntimeError(
            "Output already contains generated files. "
            "Use a clean full_batch_draft directory "
            "or archive the previous run first."
        )

    print("===================================")
    print("FULL DATASET ANNOTATION STARTED")
    print("===================================")

    print("Images:", len(image_folders))
    print("Tiles:", len(all_jobs))
    print("Output:", OUTPUT_DIR)

    summaries = []
    review_rows = []

    for number, (image_id, tile_path, metadata) in enumerate(
        all_jobs,
        start=1
    ):

        summary, reviews = process_tile(
            image_id,
            tile_path,
            metadata
        )

        summaries.append(summary)
        review_rows.extend(reviews)

        if number % 10 == 0 or number == len(all_jobs):

            print(
                f"Processed {number}/{len(all_jobs)} tiles"
            )

    # ======================================
    # EXPORT AUDIT REPORTS
    # ======================================

    summary_path = (
        REPORT_DIR / "tile_summary.csv"
    )

    review_path = (
        REPORT_DIR / "candidate_review.csv"
    )

    write_csv(
        summary_path,
        summary_fields,
        summaries
    )

    write_csv(
        review_path,
        CSV_FIELDS,
        review_rows
    )

    total_blobs = sum(
        row["exported_blobs"]
        for row in summaries
    )

    total_streaks = sum(
        row["exported_streaks"]
        for row in summaries
    )

    total_conflicts = sum(
        row["conflicts"]
        for row in summaries
    )

    total_edges = sum(
        row["edge_flags"]
        for row in summaries
    )

    total_rejected = sum(
        row["rejected_polygons"]
        for row in summaries
    )

    print("\n===================================")
    print("FULL BATCH COMPLETED")
    print("===================================")

    print("Tiles processed:", len(summaries))
    print("Draft blob polygons:", total_blobs)
    print("Reviewed streak polygons:", total_streaks)
    print("Blob/streak conflicts:", total_conflicts)
    print("Boundary flags:", total_edges)
    print("Polygon rejections:", total_rejected)
    print("Review records:", len(review_rows))

    print("\nOutput:", OUTPUT_DIR)
    print("Summary:", summary_path)
    print("Review report:", review_path)

    print("\nAll annotations remain DRAFT.")
    print("Original FITS data was not modified.")


if __name__ == "__main__":
    main()
