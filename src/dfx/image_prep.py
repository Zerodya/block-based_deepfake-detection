"""Shared image preparation, used by BOTH the real and the generated pipeline.

-------------------------------------------------------------------------------
Why this is shared
-------------------------------------------------------------------------------
The two halves of the dataset used to be written by different code with
different settings: real images came from ImageNet (already JPEG), were centre
cropped and re-saved as JPEG quality 95, while generated images were written
straight out of the sampler as lossless PNG. Every real image therefore carried
8x8 JPEG quantisation artifacts and every generated one carried none.

That is a shortcut: a classifier can separate the two classes from a global,
spatially uniform statistic without ever looking at content. And when the
discriminative cue is global, there is no region for a saliency map to point at,
so the attribution maps are legitimately flat - which is most of the reason the
explainability results were uninterpretable.

Routing both classes through this one function makes encoding history, crop
policy and output format identical by construction, so the only thing left to
separate the classes is the image content itself.
"""

import io
import random
from pathlib import Path

from PIL import Image

__all__ = ['process_image', 'random_jpeg_history',
           'DEFAULT_SIZE', 'DEFAULT_QUALITY', 'DEFAULT_FORMAT',
           'DEFAULT_HISTORY_RANGE']

DEFAULT_SIZE = 1024
DEFAULT_QUALITY = 95
DEFAULT_FORMAT = 'JPEG'
DEFAULT_HISTORY_RANGE = (70, 95)


def random_jpeg_history(img, rng=None, qrange=DEFAULT_HISTORY_RANGE):
    """Give an image a randomised prior JPEG generation.

    Matching the OUTPUT format of the two classes is necessary but not
    sufficient. The real images come from ImageNet, which is JPEG already, so
    they carry compression artifacts that survive any later re-encode; the
    generated images start pristine. Measured on a controlled pair where content
    was identical by construction, matching only the output format left a
    global-statistics probe at 100% accuracy and a noise-residual probe at 100%.
    Giving BOTH classes the same randomised prior JPEG generation dropped those
    to 74% and 70%.

    So this is the step that actually removes the compression shortcut, and it
    belongs in dataset preparation. `RandomJPEG` in dfx.dataset_classes does the
    same thing again at training time, which additionally stops the model
    keying on any residual difference in artifact strength.
    """
    rng = rng or random
    buffer = io.BytesIO()
    img.convert('RGB').save(buffer, format='JPEG',
                            quality=rng.randint(*qrange))
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def process_image(src, dst, size=DEFAULT_SIZE, crop_mode='center',
                  quality=DEFAULT_QUALITY, image_format=DEFAULT_FORMAT,
                  rng=None, jpeg_history=True,
                  history_range=DEFAULT_HISTORY_RANGE):
    """Open, convert to RGB, resize if needed, crop to size x size, save.

    `src` may be a path or an already-loaded PIL image, so the generation script
    can feed a freshly sampled image straight through without a lossless
    intermediate write that would reintroduce the asymmetry.

    `jpeg_history` is the setting that actually closes the compression
    shortcut; see random_jpeg_history() for the measurements.

    crop_mode 'random' is worth preferring over 'center' for the real half:
    ImageNet images often carry a watermark or caption in a fixed corner, and a
    centre crop keeps that overlay in the same place in every single image,
    which is itself a positional shortcut.
    """
    rng = rng or random
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = img.convert('RGB')

    # Applied BEFORE the crop and the final encode, so it stands in for the
    # capture-and-upload history a real photograph has and a fresh render does
    # not. Pass jpeg_history=False only to reproduce the original dataset.
    if jpeg_history:
        img = random_jpeg_history(img, rng, history_range)

    w, h = img.size
    if w < size or h < size:
        ratio = size / min(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    w, h = img.size
    if crop_mode == 'center':
        left, top = (w - size) // 2, (h - size) // 2
    elif crop_mode == 'random':
        left = rng.randint(0, max(0, w - size))
        top = rng.randint(0, max(0, h - size))
    else:
        raise ValueError(f'crop_mode non valido: {crop_mode}')

    img = img.crop((left, top, left + size, top + size))

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if image_format.upper() == 'JPEG':
        img.save(dst, format='JPEG', quality=quality, optimize=True)
    else:
        img.save(dst, format=image_format.upper())
    return dst
