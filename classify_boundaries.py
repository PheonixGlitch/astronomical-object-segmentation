
from pathlib import Path
from collections import Counter
import csv
import json

ROOT = Path(__file__).resolve().parent

TILES_DIR = ROOT / "outputs" / "tiles"
AUDIT_PATH = (
    ROOT / "outputs" / "reconstructed_annotations"
    / "reconstruction_audit.csv"
)

OUT = (
    ROOT / "outputs"
    / "reconstructed_annotations"
    / "boundary_classification.csv"
)

metadata_cache = {}

def get_metadata(image_id):

    if image_id not in metadata_cache:

        path = (
            TILES_DIR / image_id / "metadata.json"
        )

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata_cache[image_id] = {
            "height": int(metadata["source_shape"][0]),
            "width": int(metadata["source_shape"][1]),
            "tiles": {
                tile["filename"]: tile
                for tile in metadata["tiles"]
            }
        }

    return metadata_cache[image_id]


results = []
counts = Counter()

with AUDIT_PATH.open(
    "r", newline="", encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["status"] != "BOUNDARY_REVIEW":
            continue

        image_id = row["image"]
        tile_name = row["tile"]

        metadata = get_metadata(image_id)
        tile = metadata["tiles"][tile_name]

        x0 = int(tile["x_start"])
        y0 = int(tile["y_start"])

        w = int(tile["valid_width"])
        h = int(tile["valid_height"])

        image_w = metadata["width"]
        image_h = metadata["height"]

        flags = row["edge_flags"].split(";")

        internal = []
        external = []

        for edge in flags:

            if edge == "left":
                (external if x0 == 0 else internal).append(edge)

            elif edge == "right":
                (
                    external
                    if x0 + w == image_w
                    else internal
                ).append(edge)

            elif edge == "top":
                (external if y0 == 0 else internal).append(edge)

            elif edge == "bottom":
                (
                    external
                    if y0 + h == image_h
                    else internal
                ).append(edge)

        if internal and external:
            category = "BOTH"
        elif internal:
            category = "INTERNAL"
        elif external:
            category = "EXTERNAL"
        else:
            category = "UNCLASSIFIED"

        counts[category] += 1

        results.append({
            "image": image_id,
            "tile": tile_name,
            "line_number": row["line_number"],
            "class_id": row["class_id"],
            "category": category,
            "internal_edges": ";".join(internal),
            "external_edges": ";".join(external)
        })


with OUT.open(
    "w", newline="", encoding="utf-8"
) as f:

    fields = [
        "image",
        "tile",
        "line_number",
        "class_id",
        "category",
        "internal_edges",
        "external_edges"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)


print("\nBOUNDARY CLASSIFICATION COMPLETE")
print("--------------------------------")

for category in [
    "INTERNAL",
    "EXTERNAL",
    "BOTH",
    "UNCLASSIFIED"
]:
    print(f"{category}: {counts[category]}")

print("Total flagged polygons:", len(results))
print("Report:", OUT)
print("Existing annotations unchanged.")
