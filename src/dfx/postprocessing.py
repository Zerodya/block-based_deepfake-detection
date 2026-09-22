"""Social-media style post-processing transforms, and tensor/PIL conversion.

These simulate what happens to an image between capture/generation and the point
where a detector sees it: recompression, rescaling, screenshotting, and the
brightness/contrast/sharpness/saturation edits a platform or a user applies.

-------------------------------------------------------------------------------
Design note: these are PIL -> PIL
-------------------------------------------------------------------------------
The previous versions were tensor -> tensor, which hid a normalisation
asymmetry: `pil_to_tensor` ALWAYS applied ImageNet normalisation, while
`tensor_to_pil` only denormalised when it saw values outside [0, 1]. A round trip
happened to come out right, but only by accident, and feeding an already-[0,1]
tensor through the pair silently produced a normalised tensor. Operating on PIL
images removes the ambiguity: there is exactly one representation, and callers
normalise once with their own training transform.
"""

import io

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

__all__ = [
    'jpeg_compression',
    'gaussian_blur',
    'resize_down_up',
    'screenshot_simulation',
    'brightness_adjust',
    'contrast_adjust',
    'sharpness_adjust',
    'color_adjust',
    'tensor_to_pil',
    'pil_to_tensor',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == 'RGB' else img.convert('RGB')


# ============================================================================
# Transforms
# ============================================================================
def jpeg_compression(quality: int = 75):
    """Re-encode through JPEG at the given quality factor."""
    def transform(img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        _as_rgb(img).save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    return transform


def gaussian_blur(radius: float = 2.0):
    def transform(img: Image.Image) -> Image.Image:
        return _as_rgb(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return transform


def resize_down_up(scale: float = 0.5):
    """Downscale then restore the original size, losing detail on the way."""
    def transform(img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        w, h = img.size
        small = img.resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    return transform


def screenshot_simulation(noise_std: float = 3.0, quality: int = 85, seed=None):
    """Resample + sensor-ish noise + JPEG, as if the image had been screenshotted.

    `seed` matters: the previous version drew from the global `np.random` with no
    seed, which made every consistency score computed against it irreproducible.
    """
    def transform(img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        w, h = img.size
        displayed = img.resize((w, h), Image.BILINEAR)

        rng = np.random.default_rng(seed)
        arr = np.asarray(displayed).astype(np.float32)
        noisy = np.clip(arr + rng.normal(0.0, noise_std, arr.shape), 0, 255).astype(np.uint8)

        buffer = io.BytesIO()
        Image.fromarray(noisy).save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    return transform


def _enhancer(enhancer_cls, factor):
    def transform(img: Image.Image) -> Image.Image:
        return enhancer_cls(_as_rgb(img)).enhance(factor)
    return transform


def brightness_adjust(factor: float = 1.2):
    return _enhancer(ImageEnhance.Brightness, factor)


def contrast_adjust(factor: float = 1.3):
    return _enhancer(ImageEnhance.Contrast, factor)


def sharpness_adjust(factor: float = 2.0):
    return _enhancer(ImageEnhance.Sharpness, factor)


def color_adjust(factor: float = 1.2):
    """Saturation."""
    return _enhancer(ImageEnhance.Color, factor)


# ============================================================================
# Conversion
# ============================================================================
def tensor_to_pil(tensor: torch.Tensor, normalized: bool = None) -> Image.Image:
    """Tensor (1,C,H,W) or (C,H,W) -> PIL RGB.

    `normalized` states whether the tensor carries ImageNet normalisation. Pass
    it explicitly. The default (None) falls back to the old range heuristic,
    which mis-detects a normalised tensor whose values happen to land inside
    [0, 1] and is kept only for backwards compatibility.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy()

    if normalized is None:
        normalized = bool(img.min() < 0 or img.max() > 1)
    if normalized:
        mean = np.array(IMAGENET_MEAN).reshape(-1, 1, 1)
        std = np.array(IMAGENET_STD).reshape(-1, 1, 1)
        img = np.clip(img * std + mean, 0, 1)

    img = np.transpose(img, (1, 2, 0))
    return Image.fromarray(np.uint8(np.clip(img * 255, 0, 255)))


def pil_to_tensor(img: Image.Image, device=None, normalize: bool = True) -> torch.Tensor:
    """PIL RGB -> tensor (1,3,H,W). Set `normalize=False` to stay in [0, 1]."""
    arr = np.asarray(_as_rgb(img)).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    if normalize:
        mean = np.array(IMAGENET_MEAN).reshape(-1, 1, 1)
        std = np.array(IMAGENET_STD).reshape(-1, 1, 1)
        arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor.to(device) if device is not None else tensor
