
from pathlib import Path
from collections import Counter, defaultdict
import csv

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

VALIDATION = (
    ROOT / "outputs"
    / "boundary_merging"
    / "all_merge_validation.csv"
)

LABELS = (
    ROOT / "outputs"
    / "full_batch_draft"
    / "labels"
)

OUT = ROOT / "outputs" / "boundary_merging"
OUT.mkdir(parents=True, exist_ok=True)

DECISION_FILE = OUT / "merge_decisions.csv"

# Visual review decisions
HOLD_PAIRS = {17, 21, 78}
REJECT_PAIRS = {84, 85, 86}

EXPECTED_PAIRS = set(range(1, 87))
EXPECTED_APPROVED = 80

# ==========================================
# 2. READ VALIDATION REPORT
# ==========================================

with VALIDATION.open(
    "r", newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

pair_ids = [int(r["pair"]) for r in rows]

if len(pair_ids) != len(set(pair_ids)):
    raise ValueError("Duplicate pair IDs detected")

if set(pair_ids) != EXPECTED_PAIRS:
    raise ValueError(
        "Candidate IDs do not match the 86 reviewed pairs"
    )

# ==========================================
# 3. ASSIGN REVIEW DECISIONS
# ==========================================

decisions = []

for row in rows:
    pair_id = int(row["pair"])

    if pair_id in HOLD_PAIRS:
        decision = "HOLD"
        reason = "Additional source or class review"

    elif pair_id in REJECT_PAIRS:
        decision = "REJECT"
        reason = "Disconnected or unrelated sources"

    else:
        decision = "PROVISIONAL_APPROVE"
        reason = "Visual review and geometry support merge"

    record = dict(row)
    record["decision"] = decision
    record["decision_reason"] = reason

    decisions.append(record)

# ==========================================
# 4. VERIFY ORIGINAL ANNOTATION REFERENCES
# ==========================================

usage = defaultdict(list)
errors = []

for row in decisions:

    if row["decision"] != "PROVISIONAL_APPROVE":
        continue

    pair_id = int(row["pair"])

    if (
        row["status"] not in (
            "UNIQUE_GEOMETRY_PASS_REVIEW",
            "SHARED_FRAGMENT_REVIEW"
        )
        or int(row["components"]) != 1
        or float(row["mask_iou"]) < 0.95
    ):
        errors.append(
            f"Pair {pair_id}: geometry validation failed"
        )
        continue

    classes = []

    for tile_field, line_field in (
        ("source_tile", "source_line"),
        ("neighbor_tile", "neighbor_line")
    ):

        tile = row[tile_field]
        line_number = int(row[line_field])

        path = LABELS / (
            f"{row['image']}_{Path(tile).stem}.txt"
        )

        if not path.exists():
            errors.append(f"Missing label file: {path}")
            continue

        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

        if not 1 <= line_number <= len(lines):
            errors.append(
                f"Pair {pair_id}: invalid line {line_number}"
            )
            continue

        parts = lines[line_number - 1].split()

        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            errors.append(
                f"Pair {pair_id}: malformed annotation"
            )
            continue

        if len(coords) < 6 or len(coords) % 2:
            errors.append(
                f"Pair {pair_id}: invalid polygon vertices"
            )
            continue

        if not all(
            0.0 <= v <= 1.0 for v in coords
        ):
            errors.append(
                f"Pair {pair_id}: invalid coordinates"
            )
            continue

        classes.append(class_id)

        fragment_id = (
            row["image"], tile, line_number
        )

        usage[fragment_id].append(pair_id)

    if len(classes) != 2:
        errors.append(
            f"Pair {pair_id}: missing source fragment"
        )
    elif classes[0] != classes[1]:
        errors.append(
            f"Pair {pair_id}: class mismatch"
        )
    elif classes[0] != int(row["class_id"]):
        errors.append(
            f"Pair {pair_id}: report class mismatch"
        )

# Check for duplicate use in approved pairs
for fragment, pair_list in usage.items():
    if len(pair_list) > 1:
        errors.append(
            f"Fragment {fragment} reused by "
            f"approved pairs {pair_list}"
        )

counts = Counter(
    row["decision"] for row in decisions
)

if counts["PROVISIONAL_APPROVE"] != EXPECTED_APPROVED:
    errors.append("Unexpected approval count")

# ==========================================
# 5. SAVE DECISION REPORT
# ==========================================

fields = list(decisions[0].keys())

with DECISION_FILE.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f, fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(decisions)

# ==========================================
# 6. SUMMARY
# ==========================================

print("\n===================================")
print("MERGE DECISION PREPARATION COMPLETE")
print("===================================")

print("Candidate pairs:", len(decisions))
print(
    "Provisionally approved:",
    counts["PROVISIONAL_APPROVE"]
)
print("Held for review:", counts["HOLD"])
print("Rejected:", counts["REJECT"])

print("Unique approved fragments:", len(usage))
print("Validation errors:", len(errors))

for error in errors:
    print("ERROR:", error)

print("\nDecision CSV:", DECISION_FILE)
print("Original YOLO annotations unchanged.")

if errors:
    raise RuntimeError(
        "Validation errors found. Do not merge."
    )

print("\nReady for reconciliation preview.")
