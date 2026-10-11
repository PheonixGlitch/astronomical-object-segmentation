
from pathlib import Path
import csv
import json
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

TILES_DIR = ROOT / "outputs" / "tiles"
LABEL_DIR = ROOT / "outputs" / "full_batch_draft" / "labels"

OUT = ROOT / "outputs" / "reconstructed_annotations"
OUT.mkdir(parents=True, exist_ok=True)

FULL_LABEL_DIR = OUT / "labels"
FULL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1024
EPS = 1e-4

audit_rows = []
summary_rows = []

total_polygons = 0
total_edge_flags = 0


def parse_polygon(line):
    parts = line.split()

    if len(parts) < 7 or len(parts[1:]) % 2:
        raise ValueError("Invalid polygon format")

    class_id = int(parts[0])

    if class_id not in (0, 1):
        raise ValueError("Unexpected class ID")

    values = [float(v) for v in parts[1:]]

    if any(not (0 <= v <= 1) for v in values):
        raise ValueError("Invalid normalized coordinate")

    points = [
        (values[i], values[i + 1])
        for i in range(0, len(values), 2)
    ]

    return class_id, points


def reconstruct_polygon(points, tile, source_shape):
    height, width = source_shape

    x_start = int(tile["x_start"])
    y_start = int(tile["y_start"])

    valid_width = int(tile["valid_width"])
    valid_height = int(tile["valid_height"])

    global_points = []
    edge_flags = set()

    for x_norm, y_norm in points:

        local_x = x_norm * TILE_SIZE
        local_y = y_norm * TILE_SIZE

        if (
            local_x < -EPS
            or local_y < -EPS
            or local_x > valid_width + EPS
            or local_y > valid_height + EPS
        ):
            raise ValueError(
                "Polygon exceeds valid tile dimensions"
            )

        if local_x <= EPS:
            edge_flags.add("left")

        if local_y <= EPS:
            edge_flags.add("top")

        if local_x >= valid_width - EPS:
            edge_flags.add("right")

        if local_y >= valid_height - EPS:
            edge_flags.add("bottom")

        global_x = x_start + local_x
        global_y = y_start + local_y

        if not (
            -EPS <= global_x <= width + EPS
            and -EPS <= global_y <= height + EPS
        ):
            raise ValueError(
                "Polygon exceeds original image"
            )

        global_points.append((
            max(0.0, min(global_x, width)),
            max(0.0, min(global_y, height))
        ))

    return global_points, sorted(edge_flags)


def to_full_yolo(class_id, points, source_shape):
    height, width = source_shape

    values = [str(class_id)]

    for x, y in points:
        values.append(f"{x / width:.10f}")
        values.append(f"{y / height:.10f}")

    return " ".join(values)


def process_image(folder):
    global total_polygons
    global total_edge_flags

    image_id = folder.name

    metadata_path = folder / "metadata.json"

    with metadata_path.open(
        "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    source_shape = tuple(metadata["source_shape"])

    if len(source_shape) != 2:
        raise ValueError("Invalid source shape")

    height, width = map(int, source_shape)
    source_shape = (height, width)

    output_lines = []
    polygon_count = 0
    edge_count = 0

    tiles = metadata["tiles"]

    for tile in tiles:
        filename = tile["filename"]
        stem = Path(filename).stem

        label_path = (
            LABEL_DIR / f"{image_id}_{stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(lines, 1):

            if not line.strip():
                continue

            class_id, points = parse_polygon(line)

            global_points, edge_flags = (
                reconstruct_polygon(
                    points,
                    tile,
                    source_shape
                )
            )

            output_lines.append(
                to_full_yolo(
                    class_id,
                    global_points,
                    source_shape
                )
            )

            polygon_count += 1
            total_polygons += 1

            if edge_flags:
                edge_count += 1
                total_edge_flags += 1

            audit_rows.append({
                "image": image_id,
                "source_fits": metadata["source_filename"],
                "tile": filename,
                "line_number": line_number,
                "class_id": class_id,
                "edge_flags": ";".join(edge_flags),
                "status": (
                    "BOUNDARY_REVIEW"
                    if edge_flags else "RECONSTRUCTED"
                )
            })

    # Save full-image normalized polygons.
    output_path = (
        FULL_LABEL_DIR / f"{image_id}.txt"
    )

    output_path.write_text(
        "\n".join(output_lines)
        + ("\n" if output_lines else ""),
        encoding="utf-8"
    )

    summary_rows.append({
        "image": image_id,
        "source_width": width,
        "source_height": height,
        "tiles": len(tiles),
        "polygons": polygon_count,
        "boundary_flags": edge_count
    })

    print(
        f"{image_id}: "
        f"{len(tiles)} tiles | "
        f"{polygon_count} polygons | "
        f"{edge_count} boundary flags"
    )


def save_csv(path, rows, fields):
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


def main():
    folders = sorted(
        folder for folder in TILES_DIR.iterdir()
        if folder.is_dir()
        and re.fullmatch(r"image_\d+", folder.name)
    )

    if len(folders) != 10:
        raise ValueError(
            f"Expected 10 image folders, found {len(folders)}"
        )

    print("Starting full-image reconstruction...\n")

    for folder in folders:
        process_image(folder)

    save_csv(
        OUT / "reconstruction_audit.csv",
        audit_rows,
        [
            "image",
            "source_fits",
            "tile",
            "line_number",
            "class_id",
            "edge_flags",
            "status"
        ]
    )

    save_csv(
        OUT / "reconstruction_summary.csv",
        summary_rows,
        [
            "image",
            "source_width",
            "source_height",
            "tiles",
            "polygons",
            "boundary_flags"
        ]
    )

    total_tiles = sum(
        row["tiles"] for row in summary_rows
    )

    if total_tiles != 700:
        raise ValueError(
            f"Expected 700 tiles, found {total_tiles}"
        )

    print("\n================================")
    print("RECONSTRUCTION COMPLETED")
    print("================================")

    print("Images:", len(folders))
    print("Tiles:", total_tiles)
    print("Total polygons:", total_polygons)
    print("Boundary flags:", total_edge_flags)
    print("Output:", OUT)
    print("\nAll reconstructed labels remain DRAFT.")
    print("Original tile annotations unchanged.")


if __name__ == "__main__":
    main()
