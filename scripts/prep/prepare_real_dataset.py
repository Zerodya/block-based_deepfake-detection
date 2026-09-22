#!/usr/bin/env python3
import argparse
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict

from PIL import Image

# Shared with the generation pipeline so both halves of the dataset get an
# identical crop policy, encoding and output format. See src/dfx/image_prep.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from dfx.image_prep import process_image

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


def process_images(selected: list[Path], dest: Path, keep_structure: bool,
                   root: Path, size: int, crop_mode: str, quality: int,
                   jpeg_history: bool = True):
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
            process_image(src, dst, size=size, crop_mode=crop_mode,
                          quality=quality, jpeg_history=jpeg_history)
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
        "--crop-mode", choices=["center", "random"], default="random",
        help="Modalita' crop (default: random). 'center' keeps any watermark or "
             "caption in the same place in every image, which is itself a "
             "positional shortcut the detector can learn."
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
        "--no-jpeg-history", action="store_true",
        help="Skip the randomised prior JPEG generation. Both halves of the "
             "dataset must use the same setting, or compression history alone "
             "separates the classes."
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
        args.size, args.crop_mode, args.quality,
        jpeg_history=not args.no_jpeg_history
    )

    logger.info("Fatto!")


if __name__ == "__main__":
    main()