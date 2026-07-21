#!/usr/bin/env python3
"""
Estrae N immagini casuali da ImageNet, le croppa a 1024x1024
e le salva in una directory di destinazione.

python sample_imagenet.py -s /path/to/train -d ./mio_campione -n 1000
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict

from PIL import Image

# --- Configurazione logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def find_images(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        logger.error(f"Directory sorgente non esiste: {root}")
        sys.exit(1)

    logger.info(f"Scansione di: {root}")
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]

    if not images:
        logger.error("Nessuna immagine trovata.")
        sys.exit(1)

    logger.info(f"Trovate {len(images)} immagini.")
    return images


def sample_images(images: list[Path], n: int, balanced: bool) -> list[Path]:
    total = len(images)
    if n > total:
        logger.warning(f"Richieste {n} immagini, ne esistono solo {total}. Prendo tutte.")
        n = total

    if not balanced:
        return random.sample(images, n)

    by_folder = defaultdict(list)
    for img in images:
        by_folder[img.parent].append(img)

    folders = list(by_folder.keys())
    num_folders = len(folders)
    base = n // num_folders
    extra = n % num_folders

    selected = []
    random.shuffle(folders)

    for i, folder in enumerate(folders):
        count = base + (1 if i < extra else 0)
        pool = by_folder[folder]
        if count > len(pool):
            logger.warning(
                f"Cartella {folder.name}: solo {len(pool)} img, richieste {count}. Prendo tutte."
            )
            selected.extend(pool)
        else:
            selected.extend(random.sample(pool, count))

    random.shuffle(selected)
    return selected[:n]


def process_image(src: Path, dst: Path, size: int, crop_mode: str, quality: int):
    """
    Apre l'immagine, la converte in RGB, fa resize se necessario,
    croppa a size x size e salva in JPEG.
    """
    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size

        # Se troppo piccola, resize up mantenendo aspect ratio
        # in modo che il lato più corto sia almeno 'size'
        if w < size or h < size:
            ratio = size / min(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Crop
        w, h = img.size
        if crop_mode == "center":
            left = (w - size) // 2
            top = (h - size) // 2
        elif crop_mode == "random":
            left = random.randint(0, max(0, w - size))
            top = random.randint(0, max(0, h - size))
        else:
            raise ValueError(f"crop_mode non valido: {crop_mode}")

        right = left + size
        bottom = top + size
        img = img.crop((left, top, right, bottom))

        # Salva
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=quality, optimize=True)


def process_images(selected: list[Path], dest: Path, keep_structure: bool,
                   root: Path, size: int, crop_mode: str, quality: int):
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing di {len(selected)} immagini -> {dest}")

    ok = 0
    fail = 0

    for src in selected:
        # Forza estensione .jpg in uscita
        name = src.stem + ".jpg"

        if keep_structure:
            rel = src.relative_to(root).parent
            dst = dest / rel / name
        else:
            dst = dest / name

        # Evita collisioni se appiattito
        if not keep_structure:
            counter = 1
            original_dst = dst
            while dst.exists():
                dst = dest / f"{src.stem}_{counter:04d}.jpg"
                counter += 1

        try:
            process_image(src, dst, size, crop_mode, quality)
            ok += 1
        except Exception as e:
            logger.error(f"Errore processando {src}: {e}")
            fail += 1

    logger.info(f"Completato: {ok} salvate, {fail} errori.")


def main():
    parser = argparse.ArgumentParser(
        description="Campiona immagini da ImageNet e le croppa a 1024x1024"
    )
    parser.add_argument(
        "--src", "-s", type=Path,
        default=Path("imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"),
        help="Directory sorgente"
    )
    parser.add_argument(
        "--dest", "-d", type=Path,
        default=Path("imagenet_sample_5000"),
        help="Directory destinazione"
    )
    parser.add_argument(
        "-n", type=int, default=5000,
        help="Numero di immagini da campionare"
    )
    parser.add_argument(
        "--balanced", "-b", action="store_true",
        help="Campionamento bilanciato per categoria"
    )
    parser.add_argument(
        "--keep-structure", "-k", action="store_true",
        help="Mantieni struttura sottocartelle"
    )
    parser.add_argument(
        "--size", type=int, default=1024,
        help="Dimensione crop (default: 1024)"
    )
    parser.add_argument(
        "--crop-mode", choices=["center", "random"], default="center",
        help="Modalità crop: center o random (default: center)"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="Qualità JPEG di output (default: 95)"
    )
    parser.add_argument(
        "--ext", nargs="+",
        default=[".jpeg", ".jpg", ".JPEG", ".JPG"],
        help="Estensioni da considerare"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed per riproducibilità"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Seed: {args.seed}")

    images = find_images(args.src, tuple(args.ext))
    selected = sample_images(images, args.n, args.balanced)
    process_images(
        selected, args.dest, args.keep_structure, args.src,
        args.size, args.crop_mode, args.quality
    )

    logger.info("Fatto!")


if __name__ == "__main__":
    main()