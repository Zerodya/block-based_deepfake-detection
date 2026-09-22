#!/usr/bin/env python3
"""Measure how much of the real-vs-generated task is solvable without content.

Why this exists. The attribution maps for this detector are flat and
image-independent. One explanation is a broken saliency method; the other is
that the model separates the classes with a GLOBAL statistic, in which case
there is no region for a map to point at and a flat map is the correct answer.
This script decides between them with numbers instead of pictures, and it does
so without training the real model at all.

Four probes, cheapest first. Each reports the accuracy of a classifier that is
DENIED the thing you hope the detector is using:

  P1 container    - file suffix and JPEG quantisation table only. No pixels are
                    read. If this separates the classes, the task is trivial.
  P2 global stats - ~16 numbers per image (channel moments, DCT high-frequency
                    energy, 8x8 blockiness, noise-residual spread, saturation).
                    Logistic regression. No spatial information whatsoever.
  P3 thumbnail    - the image crushed to 32x32, destroying every high-frequency
                    artifact. Isolates a content/colour shortcut.
  P4 residual     - image minus a median filter, i.e. noise only, with content
                    removed. Isolates a sensor/compression shortcut.

Any probe near or above ~90% means the corresponding shortcut is present and
the headline accuracy of the detector does not measure what the thesis claims.

Usage:
  python shortcut_audit.py --dataset_dir DATASETS --classes dm_generated real
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_dir', required=True,
                   help='Root containing one subdirectory per class')
    p.add_argument('--classes', nargs=2, default=['dm_generated', 'real'],
                   help='The two class subdirectories to compare')
    p.add_argument('--n', type=int, default=400, help='Images per class')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip', default='', help='Comma-separated probe ids to skip')
    return p.parse_args()


def collect(root, class_name, n, rng):
    base = os.path.join(root, class_name)
    if not os.path.isdir(base):
        raise SystemExit(f'class directory not found: {base}')
    paths = [os.path.join(r, f)
             for r, _, fs in os.walk(base) for f in fs
             if os.path.splitext(f)[1].lower() in IMAGE_EXT]
    if not paths:
        raise SystemExit(f'no images under {base}')
    rng.shuffle(paths)
    return paths[:n]


# ============================================================================
# P1 - container and compression metadata, no pixels read
# ============================================================================
def probe_container(paths_a, paths_b, name_a, name_b):
    def describe(path):
        suffix = os.path.splitext(path)[1].lower()
        try:
            with Image.open(path) as img:
                fmt = img.format
                table = getattr(img, 'quantization', None)
                qkey = None
                if table:
                    qkey = tuple(sorted((k, int(np.sum(v))) for k, v in table.items()))
        except Exception:
            fmt, qkey = None, None
        return suffix, fmt, qkey

    rows_a = [describe(p) for p in paths_a]
    rows_b = [describe(p) for p in paths_b]

    print(f'    {name_a}: suffixes {dict(Counter(r[0] for r in rows_a))}  '
          f'formats {dict(Counter(r[1] for r in rows_a))}')
    print(f'    {name_b}: suffixes {dict(Counter(r[0] for r in rows_b))}  '
          f'formats {dict(Counter(r[1] for r in rows_b))}')

    # Best achievable accuracy of a lookup table on (suffix, format, q-table).
    counts = {}
    for row in rows_a:
        counts.setdefault(row, [0, 0])[0] += 1
    for row in rows_b:
        counts.setdefault(row, [0, 0])[1] += 1
    correct = sum(max(a, b) for a, b in counts.values())
    total = len(rows_a) + len(rows_b)

    n_q_a = len({r[2] for r in rows_a if r[2]})
    n_q_b = len({r[2] for r in rows_b if r[2]})
    print(f'    distinct JPEG quantisation tables: {name_a}={n_q_a}, {name_b}={n_q_b}')
    return correct / total


# ============================================================================
# P2 - global statistics only
# ============================================================================
def global_features(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((512, 512), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32) / 255.0
        grey = arr.mean(axis=2)
        median = np.asarray(
            im.convert('L').filter(ImageFilter.MedianFilter(3))).astype(np.float32) / 255.0

    feats = []
    for c in range(3):
        feats += [arr[..., c].mean(), arr[..., c].std()]

    # High-frequency energy ratio via the 2-D FFT magnitude spectrum.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey)))
    h, w = spectrum.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.hypot(yy - h / 2, xx - w / 2)
    lim = min(h, w) / 2
    total = spectrum.sum() + 1e-8
    feats += [float(spectrum[radius > 0.5 * lim].sum() / total),
              float(spectrum[radius > 0.75 * lim].sum() / total)]

    # 8x8 blockiness: discontinuity across block borders vs inside them.
    def blockiness(g, axis):
        d = np.abs(np.diff(g, axis=axis))
        idx = np.arange(d.shape[axis])
        on = np.take(d, idx[(idx + 1) % 8 == 0], axis=axis).mean()
        off = np.take(d, idx[(idx + 1) % 8 != 0], axis=axis).mean()
        return float(on / (off + 1e-8))
    feats += [blockiness(grey, 0), blockiness(grey, 1)]

    residual = grey - median
    feats += [float(residual.std()), float(np.abs(residual).mean())]

    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-8)
    feats += [float(sat.mean()), float(sat.std())]
    feats += [float(grey.mean()), float(grey.std())]
    return np.array(feats, dtype=np.float32)


def _logreg_accuracy(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    return float(cross_val_score(model, X, y, cv=5, scoring='accuracy').mean())


def probe_pixels(paths_a, paths_b, seed, mode):
    """P3 thumbnail / P4 noise residual, both through the same linear probe."""
    def vector(path):
        with Image.open(path) as im:
            im = im.convert('RGB')
            if mode == 'thumbnail':
                return (np.asarray(im.resize((32, 32), Image.BILINEAR))
                        .astype(np.float32).ravel() / 255.0)
            grey = im.convert('L').resize((256, 256), Image.BILINEAR)
            med = grey.filter(ImageFilter.MedianFilter(3))
            res = (np.asarray(grey).astype(np.float32)
                   - np.asarray(med).astype(np.float32))
            # Summarise the residual by its spectrum so the probe cannot use layout.
            spec = np.abs(np.fft.rfft2(res))
            return np.log1p(spec[:64, :64]).ravel()

    X = np.stack([vector(p) for p in paths_a + paths_b])
    y = np.array([0] * len(paths_a) + [1] * len(paths_b))
    return _logreg_accuracy(X, y, seed)


def main():
    args = parse_args()
    skip = {s.strip().upper() for s in args.skip.split(',') if s.strip()}
    import random
    rng = random.Random(args.seed)

    a_name, b_name = args.classes
    paths_a = collect(args.dataset_dir, a_name, args.n, rng)
    paths_b = collect(args.dataset_dir, b_name, args.n, rng)
    print(f'{a_name}: {len(paths_a)} images | {b_name}: {len(paths_b)} images')
    print('Chance accuracy = 0.500\n')

    results = {}

    if 'P1' not in skip:
        print('[P1] Container and compression metadata (no pixels read)')
        results['P1_container'] = probe_container(paths_a, paths_b, a_name, b_name)
        print(f"    accuracy = {results['P1_container']:.3f}\n")

    if 'P2' not in skip:
        print('[P2] ~16 global statistics, logistic regression, no spatial info')
        X = np.stack([global_features(p) for p in paths_a + paths_b])
        y = np.array([0] * len(paths_a) + [1] * len(paths_b))
        results['P2_global_stats'] = _logreg_accuracy(X, y, args.seed)
        print(f"    accuracy = {results['P2_global_stats']:.3f}\n")

    if 'P3' not in skip:
        print('[P3] 32x32 thumbnails (all high-frequency artifacts destroyed)')
        results['P3_thumbnail'] = probe_pixels(paths_a, paths_b, args.seed, 'thumbnail')
        print(f"    accuracy = {results['P3_thumbnail']:.3f}\n")

    if 'P4' not in skip:
        print('[P4] Noise residual only (content removed)')
        results['P4_residual'] = probe_pixels(paths_a, paths_b, args.seed, 'residual')
        print(f"    accuracy = {results['P4_residual']:.3f}\n")

    print('=' * 68)
    print('VERDICT')
    print('=' * 68)
    for name, acc in results.items():
        if acc >= 0.90:
            note = 'SHORTCUT - solvable without the intended cue'
        elif acc >= 0.70:
            note = 'partial shortcut'
        else:
            note = 'ok'
        print(f'  {name:<18} {acc:.3f}   {note}')
    print()
    print('  Any probe at or above ~0.90 means the headline detector accuracy')
    print('  does not measure diffusion-artifact detection. It also explains a')
    print('  flat attribution map: a global cue has no location to point at.')
    print('=' * 68)


if __name__ == '__main__':
    main()
