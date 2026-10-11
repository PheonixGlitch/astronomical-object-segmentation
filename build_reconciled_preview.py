
from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from scipy.ndimage import label
from skimage.draw import polygon2mask
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parent

TILES = ROOT / "outputs" / "tiles"

LABELS = (
    ROOT / "outputs" / "full_batch_draft"
    / "labels"
)

DECISIONS = (
    ROOT / "outputs" / "boundary_merging"
    / "merge_decisions.csv"
)

OUT = ROOT / "outputs" / "reconciled_preview"
OUT_LABELS = OUT / "labels"

SIZE = 1024
EXPECTED_ORIGINAL = 27788
EXPECTED_MERGES = 80

# ==========================================
# 1. LOAD SOURCE ANNOTATIONS
# ==========================================

annotations = defaultdict(list)
by_reference = {}
image_shapes = {}

for image_dir in sorted(TILES.glob("image_*")):

    image_id = image_dir.name

    with (image_dir / "metadata.json").open(
        "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    height, width = map(
        int, metadata["source_shape"]
    )

    image_shapes[image_id] = (height, width)

    for tile in metadata["tiles"]:

        filename = tile["filename"]

        stem = (
            f"{image_id}_{Path(filename).stem}"
        )

        path = LABELS / f"{stem}.txt"

        if not path.exists():
            raise FileNotFoundError(path)

        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line_text in enumerate(
            lines, start=1
        ):

            values = line_text.split()

            class_id = int(values[0])

            coords = np.array(
                [float(v) for v in values[1:]],
                dtype=np.float64
            )

            if (
                coords.size < 6
                or coords.size % 2
                or not np.all(np.isfinite(coords))
                or np.any(coords < 0)
                or np.any(coords > 1)
            ):
                raise ValueError(
                    f"Invalid annotation: {path}, "
                    f"line {line_number}"
                )

            points = coords.reshape(-1, 2) * SIZE

            points[:, 0] += int(tile["x_start"])
            points[:, 1] += int(tile["y_start"])

            reference = (
                image_id,
                filename,
                line_number
            )

            record = {
                "reference": reference,
                "class_id": class_id,
                "points": points
            }

            annotations[image_id].append(record)
            by_reference[reference] = record


original_count = sum(
    len(items) for items in annotations.values()
)

if original_count != EXPECTED_ORIGINAL:
    raise RuntimeError(
        f"Unexpected original count: {original_count}"
    )

# ==========================================
# 2. GEOMETRY FUNCTIONS
# ==========================================

def make_mask(points, bounds):

    x0, y0, x1, y1 = bounds

    local = np.column_stack([
        points[:, 1] - y0,
        points[:, 0] - x0
    ])

    return polygon2mask(
        (y1-y0, x1-x0),
        local
    )


def merge_polygons(first, second, image_id):

    points_a = first["points"]
    points_b = second["points"]

    both = np.vstack([points_a, points_b])

    height, width = image_shapes[image_id]

    x0 = max(0, int(np.floor(both[:, 0].min())) - 5)
    y0 = max(0, int(np.floor(both[:, 1].min())) - 5)

    x1 = min(
        width,
        int(np.ceil(both[:, 0].max())) + 6
    )

    y1 = min(
        height,
        int(np.ceil(both[:, 1].max())) + 6
    )

    bounds = (x0, y0, x1, y1)

    mask = (
        make_mask(points_a, bounds)
        | make_mask(points_b, bounds)
    )

    _, components = label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    if components != 1:
        raise ValueError(
            f"Merge not connected: {image_id}"
        )

    contours = find_contours(
        np.pad(mask.astype(np.uint8), 1),
        0.5
    )

    if len(contours) != 1:
        raise ValueError(
            f"Unexpected contour count: {image_id}"
        )

    contour = contours[0] - 1

    local_points = np.column_stack([
        contour[:, 1],
        contour[:, 0]
    ])

    local_points[:, 0] = np.clip(
        local_points[:, 0], 0, mask.shape[1]-1
    )

    local_points[:, 1] = np.clip(
        local_points[:, 1], 0, mask.shape[0]-1
    )

    if np.allclose(
        local_points[0], local_points[-1]
    ):
        local_points = local_points[:-1]

    rebuilt = polygon2mask(
        mask.shape,
        local_points[:, [1, 0]]
    )

    union = np.count_nonzero(mask | rebuilt)
    intersection = np.count_nonzero(mask & rebuilt)

    iou = intersection / union if union else 0.0

    if iou < 0.95:
        raise ValueError(
            f"Merge reconstruction IoU too low: {iou}"
        )

    global_points = local_points.copy()

    global_points[:, 0] += x0
    global_points[:, 1] += y0

    return global_points, iou


def make_yolo_line(class_id, points, image_id):

    height, width = image_shapes[image_id]

    normalized = points / np.array(
        [width, height], dtype=np.float64
    )

    if (
        not np.all(np.isfinite(normalized))
        or np.any(normalized < 0)
        or np.any(normalized > 1)
    ):
        raise ValueError(
            f"Invalid full-image coordinates: {image_id}"
        )

    values = [str(class_id)]

    for x, y in normalized:
        values.extend([
            f"{x:.10f}",
            f"{y:.10f}"
        ])

    return " ".join(values)


# ==========================================
# 3. APPLY APPROVED PAIRS IN MEMORY
# ==========================================

with DECISIONS.open(
    "r", newline="", encoding="utf-8"
) as f:
    decisions = list(csv.DictReader(f))

approved = [
    row for row in decisions
    if row["decision"] == "PROVISIONAL_APPROVE"
]

if len(approved) != EXPECTED_MERGES:
    raise RuntimeError(
        "Unexpected number of approved pairs"
    )

removed_references = set()
merged_annotations = defaultdict(list)
merge_audit = []

for row in approved:

    image_id = row["image"]

    source_ref = (
        image_id,
        row["source_tile"],
        int(row["source_line"])
    )

    neighbor_ref = (
        image_id,
        row["neighbor_tile"],
        int(row["neighbor_line"])
    )

    for ref in (source_ref, neighbor_ref):
        if ref not in by_reference:
            raise RuntimeError(
                f"Missing annotation reference: {ref}"
            )

        if ref in removed_references:
            raise RuntimeError(
                f"Repeated merge reference: {ref}"
            )

    first = by_reference[source_ref]
    second = by_reference[neighbor_ref]

    if first["class_id"] != second["class_id"]:
        raise RuntimeError("Class mismatch")

    merged_points, iou = merge_polygons(
        first, second, image_id
    )

    removed_references.update([
        source_ref, neighbor_ref
    ])

    merged_annotations[image_id].append({
        "class_id": first["class_id"],
        "points": merged_points
    })

    merge_audit.append({
        "pair": row["pair"],
        "image": image_id,
        "class_id": first["class_id"],
        "source_tile": row["source_tile"],
        "source_line": row["source_line"],
        "neighbor_tile": row["neighbor_tile"],
        "neighbor_line": row["neighbor_line"],
        "reconstruction_iou": round(iou, 6)
    })


# ==========================================
# 4. BUILD RECONCILED FULL-IMAGE LABELS
# ==========================================

# Validate everything before writing any output.
output_lines = {}

for image_id in sorted(image_shapes):

    lines = []

    for record in annotations[image_id]:

        if record["reference"] in removed_references:
            continue

        lines.append(
            make_yolo_line(
                record["class_id"],
                record["points"],
                image_id
            )
        )

    for record in merged_annotations[image_id]:

        lines.append(
            make_yolo_line(
                record["class_id"],
                record["points"],
                image_id
            )
        )

    output_lines[image_id] = lines


final_count = sum(
    len(lines) for lines in output_lines.values()
)

expected_count = (
    original_count
    - 2 * len(approved)
    + len(approved)
)

if final_count != expected_count:
    raise RuntimeError(
        "Final annotation count mismatch"
    )

# Never overwrite an existing reconciliation.
if OUT.exists() and any(OUT.iterdir()):
    raise RuntimeError(
        f"Output directory is not empty: {OUT}\n"
        "Inspect or rename the existing directory "
        "before running again."
    )

OUT_LABELS.mkdir(parents=True, exist_ok=True)

for image_id, lines in output_lines.items():

    output_file = OUT_LABELS / f"{image_id}.txt"

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


audit_file = OUT / "applied_merge_audit.csv"

with audit_file.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(merge_audit[0].keys())
    )

    writer.writeheader()
    writer.writerows(merge_audit)


# ==========================================
# 5. SUMMARY
# ==========================================

print("\n====================================")
print("RECONCILED PREVIEW CREATED")
print("====================================")

print("Original annotations:", original_count)
print("Removed fragments:", len(removed_references))
print("Merged instances added:", len(merge_audit))
print("Final annotations:", final_count)
print("Expected annotations:", expected_count)

print("\nPer-image counts:")

for image_id, lines in output_lines.items():
    print(f"{image_id}: {len(lines)}")

print("\nOutput:", OUT)
print("Merge audit:", audit_file)
print("Original annotation files unchanged.")
print("Reconciled labels are still DRAFT.")
