
from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

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
OUTPUT_DIR = ROOT / "outputs" / "batch_draft"

KNOWN_STREAK = (
    ROOT / "outputs" / "previews"
    / "merged_streak_mask.npy"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"
PREVIEW_DIR = OUTPUT_DIR / "previews"

for folder in (IMAGE_DIR, LABEL_DIR, PREVIEW_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TILE_NAMES = [
    ("image_00", "tile_r00_c00.npy"),
    ("image_04", "tile_r00_c00.npy"),
    ("image_08", "tile_r00_c00.npy")
]

SIZE = 1024

MIN_BLOB_AREA = 4
MIN_POLYGON_IOU = 0.95

# Experimental exclusion distance
STREAK_EXCLUSION_PIXELS = 3

review_rows = []


# ==========================================
# 2. CONVERT MASK TO YOLO POLYGON
# ==========================================

def mask_to_polygon(mask):
    """
    Convert a connected binary object mask
    into an XY polygon.

    Return:
        points, reconstruction_iou
    """

    padded = np.pad(
        mask.astype(np.uint8),
        1,
        mode="constant"
    )

    contours = find_contours(
        padded,
        level=0.5
    )

    if not contours:
        return None, 0.0

    contour = max(contours, key=len) - 1.0

    # Convert row/column coordinates to X/Y.
    # No additional half-pixel shift.
    points = np.column_stack([
        np.clip(contour[:, 1], 0, SIZE - 1),
        np.clip(contour[:, 0], 0, SIZE - 1)
    ])

    # Remove consecutive duplicates.
    if len(points) > 1:
        keep = np.ones(len(points), dtype=bool)

        keep[1:] = np.any(
            np.diff(points, axis=0) != 0,
            axis=1
        )

        points = points[keep]

    # Remove duplicated closing vertex.
    if len(points) > 1 and np.allclose(
        points[0], points[-1]
    ):
        points = points[:-1]

    if len(points) < 3:
        return None, 0.0

    reconstructed = polygon2mask(
        mask.shape,
        points[:, [1, 0]]
    )

    intersection = np.count_nonzero(
        mask & reconstructed
    )

    union = np.count_nonzero(
        mask | reconstructed
    )

    iou = intersection / union if union else 0.0

    return points, float(iou)


def yolo_line(class_id, points):
    """
    Format polygon in Ultralytics YOLO
    segmentation format.
    """

    normalized = points / SIZE

    values = [str(class_id)]

    for x, y in normalized:
        values.extend([
            f"{x:.8f}",
            f"{y:.8f}"
        ])

    return " ".join(values)


# ==========================================
# 3. BLOB DETECTION
# ==========================================

def detect_blobs(image, valid_mask):

    background = median_filter(
        image,
        size=31,
        mode="reflect"
    )

    residual = image - background

    valid_residual = residual[valid_mask]

    _, _, sigma = sigma_clipped_stats(
        valid_residual,
        sigma=3.0,
        maxiters=5
    )

    sigma = max(float(sigma), 1e-6)

    response = gaussian_filter(
        residual,
        sigma=1.0
    )

    binary = (
        (response > 4.0 * sigma)
        & valid_mask
    )

    components = label(
        binary,
        connectivity=2
    )

    results = []

    for region in regionprops(components):

        if region.area < MIN_BLOB_AREA:
            continue

        component_mask = (
            components == region.label
        )

        results.append(component_mask)

    return results, sigma


# ==========================================
# 4. CHECK TILE BOUNDARIES
# ==========================================

def touches_tile_boundary(
    mask,
    valid_height,
    valid_width
):

    return bool(
        np.any(mask[0, :valid_width])
        or np.any(mask[:valid_height, 0])
        or np.any(
            mask[valid_height - 1, :valid_width]
        )
        or np.any(
            mask[:valid_height, valid_width - 1]
        )
    )


# ==========================================
# 5. MAIN BATCH PROCESSING
# ==========================================

for image_id, tile_filename in TILE_NAMES:

    print(
        f"\nProcessing {image_id}/{tile_filename}"
    )

    tile_path = (
        TILES_DIR / image_id / tile_filename
    )

    metadata_path = (
        TILES_DIR / image_id / "metadata.json"
    )

    # --------------------------------------
    # 5A. LOAD IMAGE AND METADATA
    # --------------------------------------

    if not tile_path.exists():
        raise FileNotFoundError(tile_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    image = np.load(
        tile_path
    ).astype(np.float32)

    if image.shape != (SIZE, SIZE):
        raise ValueError(
            f"Unexpected dimensions: {image.shape}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    tile_info = next(
        (
            t for t in metadata["tiles"]
            if t["filename"] == tile_filename
        ),
        None
    )

    if tile_info is None:
        raise ValueError(
            f"Metadata missing for {tile_filename}"
        )

    valid_width = int(
        tile_info["valid_width"]
    )

    valid_height = int(
        tile_info["valid_height"]
    )

    if not (
        1 <= valid_width <= SIZE
        and 1 <= valid_height <= SIZE
    ):
        raise ValueError(
            "Invalid tile boundary dimensions."
        )

    valid_mask = np.zeros(
        image.shape,
        dtype=bool
    )

    valid_mask[
        :valid_height,
        :valid_width
    ] = True

    # --------------------------------------
    # 5B. DETECT BLOB CANDIDATES
    # --------------------------------------

    blobs, sigma = detect_blobs(
        image,
        valid_mask
    )

    print(
        "Initial blob candidates:",
        len(blobs)
    )

    # --------------------------------------
    # 5C. LOAD KNOWN STREAK CANDIDATE
    # --------------------------------------

    streaks = []

    if (
        image_id == "image_08"
        and tile_filename == "tile_r00_c00.npy"
    ):

        if not KNOWN_STREAK.exists():
            raise FileNotFoundError(KNOWN_STREAK)

        known = np.load(
            KNOWN_STREAK
        ).astype(bool)

        if known.shape != image.shape:
            raise ValueError(
                "Known streak mask dimensions mismatch."
            )

        known = known & valid_mask

        if np.any(known):
            streaks.append(known)

    # --------------------------------------
    # 5D. DETECT BLOB/STREAK CONFLICTS
    # --------------------------------------

    conflict_blobs = []

    if streaks:

        combined_streak_mask = np.zeros(
            image.shape,
            dtype=bool
        )

        for streak in streaks:
            combined_streak_mask |= streak

        exclusion_zone = binary_dilation(
            combined_streak_mask,
            iterations=STREAK_EXCLUSION_PIXELS
        )

        safe_blobs = []

        for blob in blobs:

            if np.any(blob & exclusion_zone):
                conflict_blobs.append(blob)
            else:
                safe_blobs.append(blob)

        blobs = safe_blobs

    print(
        "Blob/streak conflicts flagged:",
        len(conflict_blobs)
    )

    # --------------------------------------
    # 5E. PREPARE FILE NAMES
    # --------------------------------------

    stem = (
        f"{image_id}_{tile_path.stem}"
    )

    # --------------------------------------
    # 5F. SAVE GRAYSCALE IMAGE
    # --------------------------------------

    low, high = np.percentile(
        image[valid_mask],
        [1, 99.9]
    )

    high = max(float(high), float(low) + 1)

    display = np.clip(
        (image - low) / (high - low),
        0,
        1
    )

    display[~valid_mask] = 0

    plt.imsave(
        IMAGE_DIR / f"{stem}.png",
        display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    # --------------------------------------
    # 5G. CREATE ANNOTATION PREVIEW
    # --------------------------------------

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

    annotation_lines = []

    accepted_blobs = 0
    accepted_streaks = 0

    # --------------------------------------
    # 5H. SAVE CONFLICTS FOR REVIEW
    # --------------------------------------

    for conflict_index, conflict in enumerate(
        conflict_blobs
    ):

        yy, xx = np.nonzero(conflict)

        review_rows.append({
            "image": stem,
            "class_id": 0,
            "instance_index": conflict_index,
            "center_x": round(float(xx.mean()), 2),
            "center_y": round(float(yy.mean()), 2),
            "pixel_area": int(conflict.sum()),
            "polygon_iou": "",
            "exported": False,
            "review_reason": "blob_streak_conflict"
        })

        # Highlight withheld candidates in yellow.
        ax.scatter(
            [xx.mean()],
            [yy.mean()],
            color="yellow",
            marker="x",
            s=35
        )

    # --------------------------------------
    # 5I. EXPORT DRAFT YOLO POLYGONS
    # --------------------------------------

    for class_id, masks in (
        (0, blobs),
        (1, streaks)
    ):

        for instance_index, mask in enumerate(masks):

            mask = mask & valid_mask

            if not np.any(mask):
                continue

            yy, xx = np.nonzero(mask)

            center_x = float(xx.mean())
            center_y = float(yy.mean())

            boundary_flag = touches_tile_boundary(
                mask,
                valid_height,
                valid_width
            )

            # Polygon extraction assumes one component.
            component_labels = label(
                mask,
                connectivity=2
            )

            component_count = int(
                component_labels.max()
            )

            reasons = []

            if boundary_flag:
                reasons.append(
                    "touches_tile_boundary"
                )

            if component_count != 1:
                reasons.append(
                    "disconnected_instance"
                )

            points = None
            iou = 0.0

            if component_count == 1:
                points, iou = mask_to_polygon(mask)

            if points is None:
                reasons.append(
                    "polygon_conversion_failed"
                )

            elif iou < MIN_POLYGON_IOU:
                reasons.append(
                    "low_polygon_fidelity"
                )

            # Polygon must be usable before export.
            valid_polygon = (
                points is not None
                and iou >= MIN_POLYGON_IOU
                and component_count == 1
            )

            # Keep edge objects as drafts,
            # but flag them for review.
            if valid_polygon:

                annotation_lines.append(
                    yolo_line(class_id, points)
                )

                color = (
                    "cyan"
                    if class_id == 0
                    else "red"
                )

                closed = np.vstack([
                    points,
                    points[0]
                ])

                ax.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=color,
                    linewidth=1.2
                )

                if class_id == 0:
                    accepted_blobs += 1
                else:
                    accepted_streaks += 1

            review_rows.append({
                "image": stem,
                "class_id": class_id,
                "instance_index": instance_index,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "pixel_area": int(mask.sum()),
                "polygon_iou": round(iou, 4),
                "exported": valid_polygon,
                "review_reason": (
                    ";".join(reasons)
                    if reasons
                    else "manual_review_required"
                )
            })

    # --------------------------------------
    # 5J. WRITE YOLO LABEL FILE
    # --------------------------------------

    label_path = (
        LABEL_DIR / f"{stem}.txt"
    )

    label_path.write_text(
        "\n".join(annotation_lines)
        + ("\n" if annotation_lines else ""),
        encoding="utf-8"
    )

    # --------------------------------------
    # 5K. SAVE PREVIEW IMAGE
    # --------------------------------------

    ax.set_title(
        f"{stem}\n"
        f"Blobs: {accepted_blobs} | "
        f"Streaks: {accepted_streaks} | "
        f"Conflicts: {len(conflict_blobs)}"
    )

    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)

    plt.tight_layout()

    preview_path = (
        PREVIEW_DIR / f"{stem}_preview.png"
    )

    plt.savefig(
        preview_path,
        dpi=150
    )

    plt.close(fig)

    # --------------------------------------
    # 5L. TERMINAL SUMMARY
    # --------------------------------------

    print(
        f"Noise estimate: {sigma:.3f}"
    )

    print(
        "Exported blob polygons:",
        accepted_blobs
    )

    print(
        "Exported streak polygons:",
        accepted_streaks
    )

    print(
        "Conflicting blobs withheld:",
        len(conflict_blobs)
    )

    print(
        "Preview saved:",
        preview_path
    )


# ==========================================
# 6. SAVE QUALITY REVIEW REPORT
# ==========================================

report_path = (
    OUTPUT_DIR / "review_report.csv"
)

fieldnames = [
    "image",
    "class_id",
    "instance_index",
    "center_x",
    "center_y",
    "pixel_area",
    "polygon_iou",
    "exported",
    "review_reason"
]

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(review_rows)


# ==========================================
# 7. FINAL EXECUTION SUMMARY
# ==========================================

print("\n===================================")
print("BATCH ANNOTATION COMPLETED")
print("===================================")

print("Tiles processed:", len(TILE_NAMES))
print("Review records:", len(review_rows))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nQuality control report:")
print(report_path)

print("\nAll exported labels remain DRAFT candidates.")
