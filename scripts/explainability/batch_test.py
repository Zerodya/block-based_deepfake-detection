#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
import random
from pathlib import Path

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

# Regex per parsare l'output di single_test.py
PRED_RE = re.compile(r'Prediction:\s*(DM|REAL)')
PROBS_RE = re.compile(r'Probabilities:\s*DM=([\d.]+),\s*REAL=([\d.]+)')
ECS_DM_RE = re.compile(r'Mean ECS DM:\s*([\d.]+)')
ECS_REAL_RE = re.compile(r'Mean ECS REAL:\s*([\d.]+)')
STABLE_RE = re.compile(r'Stable predictions:\s*(\d+)/(\d+)')

FIELDNAMES = ['image', 'filename', 'status', 'prediction',
              'prob_dm', 'prob_real', 'mean_ecs_dm', 'mean_ecs_real',
              'stable', 'elapsed_s', 'error']


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Batch runner: esegue single_test.py su N immagini di un dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python %(prog)s --models_dir /path/to/models --backbone resnet50 \
      --dataset_dir /path/to/dataset --num_images 20

  python %(prog)s --models_dir /path/to/models --backbone resnet50 \
      --dataset_dir /path/to/dataset --num_images 50 --shuffle --seed 7
"""
    )
    parser.add_argument('--script', type=str,
                        default=str(here / 'single_test.py'),
                        help='Path di single_test.py (default: stessa cartella di questo script)')
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Cartella contenente le immagini del dataset')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Numero di immagini da processare (default: tutte)')
    parser.add_argument('--output_dir', type=str, default='../explanation_results',
                        help='Root output directory (come in single_test.py)')
    parser.add_argument('--recursive', action='store_true',
                        help='Cerca immagini anche nelle sottocartelle')
    parser.add_argument('--shuffle', action='store_true',
                        help="Mescola l'elenco prima di selezionare N immagini")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed per --shuffle (riproducibilita')")
    parser.add_argument('--timeout', type=int, default=None,
                        help='Secondi prima di terminare un run singolo (default: nessun timeout)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Mostra solo le immagini che verrebbero processate')
    parser.add_argument('--overwrite', action='store_true',
                        help='Riparte da zero: cancella il CSV esistente')
    parser.add_argument('--no_resume', action='store_true',
                        help='Riprocesso anche le immagini già OK nel CSV')
    return parser.parse_args()


def collect_images(dataset_dir, recursive):
    d = Path(dataset_dir)
    if not d.is_dir():
        raise NotADirectoryError(f'Cartella dataset non trovata: {d}')
    iterator = d.rglob('*') if recursive else d.glob('*')
    return sorted(p for p in iterator
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_done(csv_path):
    """Ritorna il set di path gia' presenti nel CSV con status OK."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('status') == 'OK':
                    done.add(row['image'])
    return done


def run_single(script, models_dir, backbone, image_path, output_dir, timeout):
    cmd = [
        sys.executable, script,
        '--models_dir', models_dir,
        '--backbone', backbone,
        '--image_path', str(image_path),
        '--output_dir', output_dir,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        stdout = proc.stdout or ''
        if proc.returncode != 0:
            return {'status': 'FAILED', 'elapsed_s': elapsed,
                    'error': (proc.stderr or stdout)[-500:]}

        pred = PRED_RE.search(stdout)
        probs = PROBS_RE.search(stdout)
        ecs_dm = ECS_DM_RE.search(stdout)
        ecs_real = ECS_REAL_RE.search(stdout)
        stable = STABLE_RE.search(stdout)

        return {
            'status': 'OK',
            'elapsed_s': f'{elapsed:.1f}',
            'prediction': pred.group(1) if pred else '',
            'prob_dm': probs.group(1) if probs else '',
            'prob_real': probs.group(2) if probs else '',
            'mean_ecs_dm': ecs_dm.group(1) if ecs_dm else '',
            'mean_ecs_real': ecs_real.group(1) if ecs_real else '',
            'stable': f'{stable.group(1)}/{stable.group(2)}' if stable else '',
        }
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'elapsed_s': f'{time.time() - t0:.1f}',
                'error': f'killed after {timeout}s'}


def main():
    args = parse_args()

    if not os.path.exists(args.script):
        sys.exit(f'ERRORE: script non trovato: {args.script}')

    images = collect_images(args.dataset_dir, args.recursive)
    if not images:
        sys.exit(f'ERRORE: nessuna immagine trovata in {args.dataset_dir}')

    if args.shuffle:
        random.Random(args.seed).shuffle(images)

    if args.num_images is not None:
        images = images[:args.num_images]

    print(f'Trovate {len(images)} immagini nel dataset: {args.dataset_dir}')
    print(f'Script: {args.script}')
    print(f'Backbone: {args.backbone} | Output: {args.output_dir}')
    print('=' * 70)

    if args.dry_run:
        for i, img in enumerate(images, 1):
            print(f'{i:>4}. {img}')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'batch_summary.csv')

    if args.overwrite and os.path.exists(csv_path):
        os.remove(csv_path)
        print(f'--overwrite: eliminato {csv_path}')

    done = set() if args.no_resume else load_done(csv_path)
    if done:
        print(f'Resume: {len(done)} immagini già OK verranno saltate '
              f'(--no_resume per disattivare, --overwrite per azzerare)')

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        n_ok = n_fail = n_skip = 0
        for i, img in enumerate(images, 1):
            if str(img) in done:
                n_skip += 1
                print(f'[{i}/{len(images)}] {img.name} ... SKIP (già OK)')
                continue

            print(f'[{i}/{len(images)}] {img.name} ...', end=' ', flush=True)
            res = run_single(args.script, args.models_dir, args.backbone,
                             img, args.output_dir, args.timeout)

            row = {'image': str(img), 'filename': img.name, 'error': res.get('error', '')}
            row.update({k: v for k, v in res.items() if k in FIELDNAMES})
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())   # garantisce che la riga sia su disco

            if res['status'] == 'OK':
                n_ok += 1
                print(f"OK -> {res['prediction']} "
                      f"(DM={res['prob_dm']}, REAL={res['prob_real']}) "
                      f"[{res['elapsed_s']}s]")
            else:
                n_fail += 1
                print(f"{res['status']}! (vedi CSV)")

    print('=' * 70)
    print(f'FINITO: {n_ok} OK, {n_fail} falliti, {n_skip} saltate '
          f'su {len(images)} immagini.')
    print(f'Summary CSV: {csv_path}')


if __name__ == '__main__':
    main()