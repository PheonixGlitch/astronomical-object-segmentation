# Astronomical Object Segmentation

Python-based astronomical image processing and segmentation pipeline developed for an AI/ML technical assessment.

## Overview

This project explores the detection and segmentation of star-like blobs and elongated streaks in FITS astronomical imagery.

The pipeline includes image tiling, background subtraction, noise estimation, source detection, streak analysis, segmentation polygon generation, boundary reconciliation, and full-image reconstruction.

## Annotation Classes

| Class | Object | Visualization |
|---|---|---|
| 0 | Star / Blob | Blue |
| 1 | Object / Streak | Red |

Annotations use the Ultralytics YOLO segmentation format:

`class_id x1 y1 x2 y2 x3 y3 ...`

Coordinates are normalized to the image dimensions.

## Implementation

**Image preprocessing:** FITS loading, 1024 × 1024 tiling, median-filter background estimation, sigma-clipped noise estimation, and Gaussian smoothing.

**Blob segmentation:** Connected-component analysis and polygon extraction from thresholded image residuals.

**Streak investigation:** Hough-based line detection, geometric measurements, and intensity-profile analysis.

**Boundary handling:** Preservation of valid pixels in padded edge tiles, followed by full-image polygon reconstruction and selected boundary-fragment reconciliation.

**Visualization:** Rendering blue and red segmentation outlines for visual inspection.

## Repository Contents

- `tile_images.py` — FITS tiling
- `batch_annotation.py` — Blob annotation pipeline
- `full_batch_annotation.py` — Batch processing
- `general_streak_detector.py` — Streak candidate investigation
- `hough_streak_detection.py` — Line detection
- `validate_streak_profiles.py` — Intensity-based streak checks
- `classify_boundaries.py` — Boundary classification
- `match_boundary_fragments.py` — Fragment matching
- `prepare_merge_decisions.py` — Boundary merge decisions
- `reconstruct_full_annotations.py` — Full-image reconstruction
- `build_reconciled_preview.py` — YOLO annotation export
- `export_full_images.py` — Full-resolution image export
- `render_annotated_images.py` — Colored polygon visualization

## Dependencies

Python, NumPy, SciPy, Astropy, OpenCV, scikit-image, Matplotlib, and Pillow.

## Reproducibility and Limitations

Scripts use project-relative paths and represent individual pipeline stages; a fully automated one-command workflow is not provided.

Detection results are algorithmically generated and require review for ambiguous objects, faint sources, and potential missed streaks. Python syntax was checked, but the public version has not undergone a complete end-to-end execution test.

## Confidentiality

This repository contains source code only. Original FITS data, generated image tiles, annotation labels, correction manifests, and proprietary assessment materials are not included.

## Author

Kevin Chris Joseph
