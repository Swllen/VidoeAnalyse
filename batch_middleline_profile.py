#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Middle-Line Profile Extractor (with smoothing) — Recursive + Peak mode

What it does, for every directory under ROOT_DIR:
  - If the directory contains images matching PATTERN:
      * load each image
      * convert to grayscale
      * extract a 1D profile (row-mean across width; length = image height)
      * moving-average smooth with WINDOW
      * choose peak index based on PEAK_MODE:
          - "first": choose the first local maximum if >=2 local maxima exist
                     (if none found, fall back to global max)
          - "max":   choose the global maximum (np.argmax)
          - "right_first": in the right-side region, take the first (from right) peak
                           whose height >= REL_HEIGHT * global_max; else rightmost local peak; else global max
          - "left_first":  symmetric to right_first on the left side
      * save a plot into "<that_directory>/profile_z/<same filename>.png"
      * annotate the chosen peak with coordinates and a vertical line

Notes:
  - If WINDOW <= 1, smoothing is bypassed (identity).
  - Index-based selection is applied per-folder, based on trailing integer in filename.
  - This script scans ALL subdirectories recursively (including ROOT_DIR itself).
"""

import os, re
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ========== EDIT THESE PARAMETERS ==========
ROOT_DIR    = r"E:\BU_Research\LabData\0901\StageTest\BPC303\Z_Channel3"   # ← root directory to scan recursively
PATTERN     = "*.png"                                     # e.g., "*.png" or "*.jpg"
WINDOW      = 20                                          # smoothing window (pixels)
DPI         = 60                                          # output figure DPI

# Peak picking mode:
#   "max"          -> global maximum
#   "first"        -> if >=2 local maxima, take the first; else global max
#   "right_first"  -> in right region (index >= SIDE_FRAC*n), pick first-from-right
#                     peak with height >= REL_HEIGHT * global_max; else rightmost local peak; else global max
#   "left_first"   -> symmetric to right_first on the left region
PEAK_MODE   = "right_first"

# Only used in right_first/left_first
REL_HEIGHT  = 0.75   # peak height threshold relative to global max (0~1)
SIDE_FRAC   = 0.55   # split position of the array (0~1). Right region starts at SIDE_FRAC*n; left ends at (1-SIDE_FRAC)*n

# ----- index selection (choose ONE or leave both empty to use all) -----
SELECT_EXPR  = ""                                         # e.g., "1-100,205,300-450:2" (takes priority)
INDEX_RANGE: Optional[Tuple[int, int]] = None             # e.g., (start, end) or None
INDEX_STEP   = 1                                          # step for INDEX_RANGE (must be > 0)
# ===========================================


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def detect_local_maxima(y: np.ndarray) -> list:
    """
    Return indices of local maxima (plateau handled by taking plateau start).
    Definition: y[i-1] < y[i] >= y[i+1];
    Plateau ...a < m==m==m > b -> take plateau start index.
    """
    n = y.size
    peaks = []
    i = 1
    while i < n - 1:
        if y[i - 1] < y[i]:
            if y[i] > y[i + 1]:
                peaks.append(i)
                i += 1
                continue
            elif y[i] == y[i + 1]:
                j = i + 1
                while j < n and y[j] == y[i]:
                    j += 1
                left_ok = y[i - 1] < y[i]
                right_ok = (j < n and y[j] < y[i])
                if left_ok and right_ok:
                    peaks.append(i)  # plateau start
                i = j
                continue
        i += 1
    return peaks


def find_peak_index(smoothed: np.ndarray, mode: str) -> int:
    """
    Modes:
      - "max"         : global maximum (first occurrence)
      - "first"       : if >=2 local maxima -> first one; else global max
      - "right_first" : right-side region [SIDE_FRAC*n, n-1]:
                        pick first (from right) local peak with y>=REL_HEIGHT*global_max,
                        else rightmost local peak in region, else global max
      - "left_first"  : left-side region [0, (1-SIDE_FRAC)*n):
                        pick first (from left) local peak with y>=REL_HEIGHT*global_max,
                        else leftmost local peak in region, else global max
    """
    y = smoothed
    n = y.size
    if n == 0:
        return 0

    mode = (mode or "max").lower().strip()
    g_idx = int(np.argmax(y))
    g_val = float(y[g_idx])

    if mode == "max":
        return g_idx

    peaks = detect_local_maxima(y)

    if mode == "first":
        if len(peaks) >= 2:
            return peaks[0]
        else:
            return g_idx

    if mode == "right_first":
        side_start = int(np.floor(SIDE_FRAC * n))
        # peaks meeting threshold in the right region
        candidates = [i for i in peaks if i >= side_start and y[i] >= REL_HEIGHT * g_val]
        if candidates:
            return max(candidates)  # first from right
        right_peaks = [i for i in peaks if i >= side_start]
        if right_peaks:
            return max(right_peaks)  # rightmost local peak
        return g_idx

    if mode == "left_first":
        side_end = int(np.ceil((1.0 - SIDE_FRAC) * n))  # left region is [0, side_end)
        candidates = [i for i in peaks if i < side_end and y[i] >= REL_HEIGHT * g_val]
        if candidates:
            return min(candidates)  # first from left
        left_peaks = [i for i in peaks if i < side_end]
        if left_peaks:
            return min(left_peaks)  # leftmost local peak
        return g_idx

    # Fallback
    return g_idx


def process_image(path: str, out_dir: str, window: int, dpi: int = 120) -> str:
    # Load and convert to grayscale
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # 1D profile along vertical axis: mean over columns -> length=h
    profile = arr.mean(axis=1)

    # Smooth
    smoothed = moving_average(profile, window)

    # ---- Choose peak index based on PEAK_MODE ----
    max_idx = find_peak_index(smoothed, PEAK_MODE)
    max_val = float(smoothed[max_idx])

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, linewidth=1.5, label=f"Smoothed (window={window})")
    # Vertical line at peak (parallel to y-axis)
    mode_label = {
        "first": "Peak (first local ≥2)",
        "max": "Peak (global max)",
        "right_first": f"Peak (right-first, ≥{int(REL_HEIGHT*100)}% of global)",
        "left_first": f"Peak (left-first, ≥{int(REL_HEIGHT*100)}% of global)",
    }.get(PEAK_MODE.lower(), "Peak")
    plt.axvline(x=max_idx, linestyle="--", label=f"{mode_label} at x={max_idx}, y={max_val:.1f}")
    # Mark the peak point
    plt.scatter([max_idx], [max_val], zorder=5)
    # Text label slightly to the right of the point
    plt.text(max_idx + max(5, h // 200), max_val, f"({max_idx}, {max_val:.1f})", va="bottom")

    plt.title(os.path.basename(path))
    plt.xlabel("Pixel Position (row index)")
    plt.ylabel("Intensity (0-255)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)

    # Keep the same filename, save as .png
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


# -------------------- index selection helpers --------------------
def extract_index(p: str) -> Optional[int]:
    """
    Extract trailing integer right before the extension.
    Example: 'frame_5955.png' -> 5955
    """
    m = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", os.path.basename(p))
    return int(m.group(1)) if m else None


def parse_selection(expr: str) -> set:
    """
    Parse expression like "1-10,25,40-50:2" into a set of integers.
    Ranges are inclusive; optional step after ':'.
    """
    sel = set()
    if not expr:
        return sel
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", p)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            step = int(m.group(3)) if m.group(3) else 1
            if step <= 0:
                raise ValueError(f"Step must be > 0: {p}")
            rng = range(start, end + 1, step) if start <= end else range(start, end - 1, -step)
            sel.update(rng)
        else:
            if not p.isdigit():
                raise ValueError(f"Cannot parse selection token: {p}")
            sel.add(int(p))
    return sel


def build_range_selection(idx_range: Optional[Tuple[int, int]], step: int) -> Optional[set]:
    if idx_range is None:
        return None
    if step <= 0:
        raise ValueError("INDEX_STEP must be > 0")
    start, end = idx_range
    if start <= end:
        return set(range(start, end + 1, step))
    else:
        return set(range(start, end - 1, -step))
# -----------------------------------------------------------------


def process_folder(folder: str) -> Tuple[int, int]:
    """
    Process one folder: returns (ok_count, fail_count)
    """
    paths: List[str] = sorted(glob(os.path.join(folder, PATTERN)))
    if not paths:
        return (0, 0)

    # Build index set from settings:
    idx_set: Optional[set] = None
    if SELECT_EXPR.strip():
        idx_set = parse_selection(SELECT_EXPR.strip())
    elif INDEX_RANGE is not None:
        idx_set = build_range_selection(INDEX_RANGE, INDEX_STEP)

    # Filter by index set if provided
    if idx_set:
        filtered = []
        missed = 0
        for p in paths:
            idx = extract_index(p)
            if idx is None:
                missed += 1
                continue
            if idx in idx_set:
                filtered.append(p)
        paths = sorted(filtered, key=lambda x: extract_index(x) or 10**18)
        if not paths:
            print(f"[{folder}] No files matched the index selection. Check SELECT_EXPR / INDEX_RANGE.")
            return (0, 0)
        if missed:
            print(f"[{folder}] Warning: {missed} files had no trailing index and were skipped.")

    out_dir = os.path.join(folder, "profile")
    print(f"[{folder}] Found {len(paths)} files. Saving outputs to: {out_dir}")

    ok, fail = 0, 0
    for p in paths:
        try:
            out = process_image(p, out_dir, WINDOW, dpi=DPI)
            print(f"  ✓ {os.path.basename(p)} -> {out}")
            ok += 1
        except Exception as e:
            print(f"  ✗ Failed on {p}: {e}")
            fail += 1

    return (ok, fail)


def run():
    root = os.path.abspath(ROOT_DIR)
    if not os.path.isdir(root):
        print(f"Root directory does not exist: {root}")
        return

    total_dirs = 0
    processed_dirs = []
    skipped_dirs = []
    total_ok = total_fail = 0

      # Walk includes ROOT_DIR itself
    skip = {"profile", "video", "videos"}  # 要跳过的目录名（不区分大小写）

    for cur_dir, subdirs, files in os.walk(root):
        # 1) 剪枝：不继续深入这些子目录
        subdirs[:] = [d for d in subdirs if d.lower() not in skip]

        # 2) 若当前目录本身就是要跳过的名字，则直接 continue
        if os.path.basename(cur_dir).lower() in skip:
            continue

        # 3) 仅当当前目录有匹配的图片才处理
        img_matches = glob(os.path.join(cur_dir, PATTERN))
        if img_matches:
            total_dirs += 1
            ok, fail = process_folder(cur_dir)
            total_ok += ok
            total_fail += fail
            processed_dirs.append(cur_dir)
        else:
            skipped_dirs.append(cur_dir)


    # ===== Summary =====
    print("\n========== SUMMARY ==========")
    print(f"Folders with images processed: {len(processed_dirs)}")
    print(f"Total images: OK {total_ok}, FAIL {total_fail}")

    if processed_dirs:
        print("\nProcessed folders:")
        for d in processed_dirs:
            print("  ✔", d)

    # You can uncomment below to list skipped folders (usually too verbose)
    # if skipped_dirs:
    #     print("\nSkipped (no matching images):")
    #     for d in skipped_dirs:
    #         print("  -", d)

    print("================================")


if __name__ == "__main__":
    run()
