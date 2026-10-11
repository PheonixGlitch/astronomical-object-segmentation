
from pathlib import Path
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parent

TILES = ROOT / "outputs" / "tiles"
LABELS = (
    ROOT / "outputs" / "full_batch_draft"
    / "labels"
)
BOUNDARIES = (
    ROOT / "outputs" / "reconstructed_annotations"
    / "boundary_classification.csv"
)

OUT = (
    ROOT / "outputs"
    / "boundary_matching"
)
OUT.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1024
MAX_ACROSS_GAP = 6.0
MAX_ALONG_GAP = 6.0

# Load metadata and annotations once.
metadata_cache = {}
polygon_cache = {}


def load_metadata(image_id):
    if image_id not in metadata_cache:
        path = TILES / image_id / "metadata.json"

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata_cache[image_id] = {
            (int(t["row"]), int(t["column"])): t
            for t in metadata["tiles"]
        }

    return metadata_cache[image_id]


def load_polygons(image_id, tile):
    filename = tile["filename"]
    key = (image_id, filename)

    if key in polygon_cache:
        return polygon_cache[key]

    stem = Path(filename).stem
    path = LABELS / f"{image_id}_{stem}.txt"

    if not path.exists():
        raise FileNotFoundError(path)

    results = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        class_id = int(parts[0])
        values = np.array(
            [float(v) for v in parts[1:]],
            dtype=float
        )

        if (
            values.size < 6
            or values.size % 2 != 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"Invalid polygon: {path}, line {line_number}"
            )

        points = values.reshape(-1, 2) * TILE_SIZE
        points[:, 0] += int(tile["x_start"])
        points[:, 1] += int(tile["y_start"])

        results.append({
            "line": line_number,
            "class_id": class_id,
            "bbox": (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max())
            )
        })

    polygon_cache[key] = results
    return results


def interval_gap(a_min, a_max, b_min, b_max):
    return max(0.0, b_min - a_max, a_min - b_max)


def candidate_distance(first, second, edge):
    ax0, ay0, ax1, ay1 = first["bbox"]
    bx0, by0, bx1, by1 = second["bbox"]

    if edge == "left":
        # First polygon is in the right-hand tile.
        across = abs(ax0 - bx1)
        along = interval_gap(ay0, ay1, by0, by1)

    elif edge == "top":
        # First polygon is in the lower tile.
        across = abs(ay0 - by1)
        along = interval_gap(ax0, ax1, bx0, bx1)

    else:
        return None

    if (
        across > MAX_ACROSS_GAP
        or along > MAX_ALONG_GAP
    ):
        return None

    return across, along


def main():
    with BOUNDARIES.open(
        "r", newline="", encoding="utf-8"
    ) as f:
        boundaries = list(csv.DictReader(f))

    matches = []
    unmatched = []
    processed = 0

    for entry in boundaries:
        if entry["category"] not in ("INTERNAL", "BOTH"):
            continue

        processed += 1

        image_id = entry["image"]
        tiles = load_metadata(image_id)

        tile = next(
            t for t in tiles.values()
            if t["filename"] == entry["tile"]
        )

        row = int(tile["row"])
        col = int(tile["column"])

        source_line = int(entry["line_number"])
        source = next(
            p for p in load_polygons(image_id, tile)
            if p["line"] == source_line
        )

        for edge in entry["internal_edges"].split(";"):
            if edge == "left":
                neighbor_key = (row, col - 1)
            elif edge == "top":
                neighbor_key = (row - 1, col)
            else:
                continue

            neighbor = tiles.get(neighbor_key)
            found = []

            if neighbor is not None:
                for target in load_polygons(image_id, neighbor):
                    if target["class_id"] != source["class_id"]:
                        continue

                    distance = candidate_distance(
                        source, target, edge
                    )

                    if distance is None:
                        continue

                    across, along = distance

                    found.append({
                        "image": image_id,
                        "source_tile": tile["filename"],
                        "source_line": source_line,
                        "neighbor_tile": neighbor["filename"],
                        "neighbor_line": target["line"],
                        "class_id": source["class_id"],
                        "edge": edge,
                        "across_gap_px": round(across, 3),
                        "along_gap_px": round(along, 3),
                        "status": "POSSIBLE_MATCH_REVIEW"
                    })

            if found:
                matches.extend(found)
            else:
                unmatched.append({
                    "image": image_id,
                    "source_tile": tile["filename"],
                    "source_line": source_line,
                    "edge": edge,
                    "status": "NO_NEARBY_POLYGON"
                })

    def save_csv(filename, rows, fields):
        with (OUT / filename).open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fields
            )
            writer.writeheader()
            writer.writerows(rows)

    save_csv(
        "possible_matches.csv",
        matches,
        [
            "image", "source_tile", "source_line",
            "neighbor_tile", "neighbor_line",
            "class_id", "edge",
            "across_gap_px", "along_gap_px", "status"
        ]
    )

    save_csv(
        "unmatched_boundaries.csv",
        unmatched,
        [
            "image", "source_tile", "source_line",
            "edge", "status"
        ]
    )

    print("\nBOUNDARY MATCHING COMPLETED")
    print("---------------------------")
    print("Internal polygons inspected:", processed)
    print("Possible matching pairs:", len(matches))
    print("Unmatched boundary sides:", len(unmatched))
    print("Reports saved:", OUT)
    print("Existing annotations unchanged.")


if __name__ == "__main__":
    main()
