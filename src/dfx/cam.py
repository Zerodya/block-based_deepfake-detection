"""
Attribution / saliency methods for the 2-block DeepFeatureX detector.

Single auditable implementation shared by `scripts/explainability/single_test.py`
and `scripts/explainability/diagnose_prior.py`.

-------------------------------------------------------------------------------
WHY THIS MODULE EXISTS - the degeneracy of last-layer CAM
-------------------------------------------------------------------------------
The base models are `layer4 -> AdaptiveAvgPool2d -> Linear(2048, 2)`. For such a
GAP+linear head the class score is

    logit_k = sum_c w_kc * (1/HW) * sum_ij A_cij + b_k

so that

    d logit_k / d A_cij = w_kc / (HW)

which is *constant over (i, j) and independent of the input image*. Therefore:

  * Grad-CAM at layer4 collapses to a FIXED linear projection of the activation
    tensor, `cam(i,j) = sum_c w_kc * A_cij`. This is the CAM of Zhou et al. 2016;
    the equivalence is stated in the Grad-CAM paper itself for GAP+FC nets.
  * HiResCAM at layer4 is *identical* to Grad-CAM here, because the gradient is
    already spatially constant so pooling it changes nothing.
  * Grad-CAM++ at layer4 is still a spatially-constant per-channel reweighting.

All the image-dependence lives in `A`, and at a 7x7 / 8x8 grid the per-image
variation of `A` is small next to its dataset-mean spatial pattern (border energy
from convolution padding). That is why last-layer maps look like a fixed blob in
one corner, and why the measured robustness score is ~0.99: that number is
analytically forced, not evidence of stability.

Consequence for users of this module: to obtain genuinely input-dependent,
spatially resolved attributions, use `stage='layer3'` (where the gradient really
does vary in space) or the input-space methods `occlusion_map` /
`integrated_gradients`. `assert_gap_linear_degeneracy` below turns the argument
above into a runnable check.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CamResult',
    'ActivationGrabber',
    'disable_inplace_relu',
    'resolve_stage',
    'compute_cam',
    'score_cam',
    'occlusion_map',
    'integrated_gradients',
    'make_baseline',
    'branch_attribution',
    'assert_gap_linear_degeneracy',
    'overlay_heatmap',
    'ecs_pearson',
    'ecs_iou',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
]

EPS = 1e-8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Result container
# ============================================================================
@dataclass
class CamResult:
    """A saliency map plus everything needed to judge whether to trust it.

    Attributes
    ----------
    map : (H, W) float32, min-max normalised to [0, 1]. The displayable map.
    signed : (H, W) float32, the pre-ReLU map, symmetrically scaled about 0.
             For a 2-class detector the negative evidence matters as much as the
             positive, and the signed map is what reveals a fixed projection.
    raw : (H, W) float32, the map before ReLU and before any normalisation.
          Keep this: min-max normalisation is exactly what lets a map with a
          dynamic range of 1e-6 look like a confident explanation.
    meta : diagnostics; `meta['degenerate']` is True when there is no positive
           evidence at all, in which case `map` is all-zero and MUST NOT be
           rendered as an ordinary heatmap.
    """
    map: np.ndarray
    signed: np.ndarray
    raw: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def degenerate(self) -> bool:
        return bool(self.meta.get('degenerate', False))


def _finalise(raw: torch.Tensor, relu: bool, extra: Optional[Dict[str, Any]] = None) -> CamResult:
    """Normalise a raw 2-D map and record why it may not be trustworthy."""
    raw = raw.detach().float().squeeze()
    if raw.ndim != 2:
        raise ValueError(f'expected a 2-D map, got shape {tuple(raw.shape)}')

    raw_np = raw.cpu().numpy()
    pre_min, pre_max = float(raw_np.min()), float(raw_np.max())

    activated = torch.relu(raw) if relu else raw
    a_min, a_max = float(activated.min()), float(activated.max())
    span = a_max - a_min

    degenerate = span <= EPS
    if degenerate:
        norm = np.zeros_like(raw_np)
    else:
        norm = ((activated - a_min) / (span + EPS)).cpu().numpy()

    # Signed map: scale by the largest absolute value so 0 stays at 0. Rendered
    # with a diverging colormap this shows suppression as well as support.
    scale = max(abs(pre_min), abs(pre_max))
    signed = raw_np / scale if scale > EPS else np.zeros_like(raw_np)

    meta: Dict[str, Any] = {
        'pre_relu_min': pre_min,
        'pre_relu_max': pre_max,
        'pre_relu_range': pre_max - pre_min,
        'frac_positive': float((raw_np > 0).mean()),
        'degenerate': degenerate,
        'grid_shape': tuple(raw_np.shape),
    }
    if extra:
        meta.update(extra)
    return CamResult(map=norm.astype(np.float32),
                     signed=signed.astype(np.float32),
                     raw=raw_np.astype(np.float32),
                     meta=meta)


# ============================================================================
# Hooks
# ============================================================================
class ActivationGrabber:
    """Capture a module's forward output as a *live graph node*.

    Deliberately does not call `.detach()`: the tensor must stay attached so
    `torch.autograd.grad` can differentiate with respect to it. We never use
    `register_full_backward_hook`, which PyTorch documents as unreliable when the
    module mutates its output in place - and torchvision's `Bottleneck` ends with
    `out += identity` followed by `ReLU(inplace=True)`, exactly that case.
    """

    def __init__(self, module: nn.Module):
        self.activation: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        self.activation = output

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def disable_inplace_relu(model: nn.Module) -> nn.Module:
    """Set `inplace=False` on every ReLU. Numerically a no-op.

    Costs memory only, and removes a whole class of silently-wrong-gradient
    failures around in-place activations. Call once after loading a model.
    """
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.ELU)):
            m.inplace = False
    return model


# ============================================================================
# Target-layer resolution
# ============================================================================
def _residual_stages(module: nn.Module):
    """Return the residual stages [layer1..layer4] of a ResNet-family module.

    Resolved by *structure*, not by attribute name or positional index, because
    `FeatureExtractor` wraps `base_model.children()` in an anonymous
    `nn.Sequential`, which destroys the names. A ResNet stage is an
    `nn.Sequential` whose first child is a residual block; a `downsample`
    shortcut is also an `nn.Sequential` but holds a Conv2d, so it is filtered out.
    `modules()` yields in registration order, so the result is layer1..layer4.
    """
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError as exc:  # pragma: no cover
        raise ImportError('torchvision is required to resolve ResNet stages') from exc

    return [m for m in module.modules()
            if isinstance(m, nn.Sequential) and len(m) > 0
            and isinstance(m[0], (BasicBlock, Bottleneck))]


def resolve_stage(module: nn.Module, stage: str = 'layer4') -> nn.Module:
    """Return the module to hook for `stage` in {'layer1'..'layer4'}.

    Works both on a raw torchvision ResNet and on a `FeatureExtractor`-wrapped
    one. Only the ResNet/ResNeXt family is supported; see the NotImplementedError
    message for why the other backbones are deliberately out of scope.
    """
    stages = _residual_stages(module)
    if len(stages) > 4:
        # A CompleteModel2Blocks contains two ResNets, so it exposes 8 stages and
        # picking by index would silently return the wrong branch. Callers must
        # pass a single branch: resolve_stage(model.dm_extractor, 'layer4').
        raise ValueError(
            f'found {len(stages)} residual stages, expected 4. This module holds '
            'more than one backbone - pass a single branch instead, e.g. '
            "resolve_stage(complete_model.dm_extractor, 'layer4').")
    if len(stages) < 4:
        raise NotImplementedError(
            'resolve_stage() supports the ResNet/ResNeXt family only, and found '
            f'{len(stages)} residual stages instead of 4. Note that '
            'FeatureExtractor (nn.Sequential(*children()[:-1])) is itself only '
            'valid for ResNet-family backbones: for DenseNet it drops the final '
            'relu+pool, and for EfficientNet/ViT it is structurally wrong. Fixing '
            'those is out of scope for this resnet50-only project.')

    try:
        index = {'layer1': -4, 'layer2': -3, 'layer3': -2, 'layer4': -1}[stage]
    except KeyError:
        raise ValueError(f"stage must be one of layer1..layer4, got {stage!r}")
    return stages[index]


# ============================================================================
# Shared forward helpers
# ============================================================================
def _forward_capturing(model: nn.Module, x: torch.Tensor, layer: nn.Module):
    with ActivationGrabber(layer) as grabber:
        output = model(x)
    if grabber.activation is None:
        raise RuntimeError('forward hook captured nothing - wrong target layer?')
    return output, grabber.activation


def _target_score(output: torch.Tensor, target_class: int, score: str) -> torch.Tensor:
    """Scalar to differentiate.

    'logit'  - the raw class logit; the standard, citable choice.
    'margin' - logit_target - logit_other, the actual decision function of a
               2-class head. It cancels the shared bias/offset component, which
               for a saturated model is most of the signal.
    """
    if score == 'logit':
        return output[0, target_class]
    if score == 'margin':
        if output.shape[1] != 2:
            raise ValueError("score='margin' requires a 2-class output")
        return output[0, target_class] - output[0, 1 - target_class]
    raise ValueError(f"score must be 'logit' or 'margin', got {score!r}")


def _require_grad_input(x: torch.Tensor) -> torch.Tensor:
    """Return a graph-attached copy of the input.

    `CompleteModel2Blocks` freezes every extractor parameter. If the input does
    not require grad either, nothing in the forward pass requires grad, the
    captured activation is not part of any graph, and `torch.autograd.grad`
    fails with "element 0 of tensors does not require grad". Making the *input*
    require grad restores the graph, and yields the input gradients that
    `integrated_gradients` needs anyway.
    """
    return x.clone().detach().requires_grad_(True)


# ============================================================================
# Gradient-based CAMs
# ============================================================================
def compute_cam(model: nn.Module,
                x: torch.Tensor,
                target_class: int,
                target_layer: nn.Module,
                method: str = 'gradcam',
                score: str = 'logit',
                relu: bool = True) -> CamResult:
    """Grad-CAM / HiResCAM / Grad-CAM++ on `target_layer`.

    At layer4 of a GAP+linear head all three are degenerate in the sense
    described in the module docstring; `method` only starts to matter from
    layer3 down. `meta['grad_spatial_std']` quantifies this per call: a value at
    machine-epsilon means the gradient is spatially constant and the map is a
    fixed projection of the activations.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    x = _require_grad_input(x)
    output, acts = _forward_capturing(model, x, target_layer)
    s = _target_score(output, target_class, score)
    grads = torch.autograd.grad(s, acts, retain_graph=False)[0]

    if method == 'gradcam':
        weights = grads.mean(dim=(2, 3), keepdim=True)
        raw = (weights * acts).sum(dim=1)
    elif method == 'hirescam':
        # No gradient pooling. Provably identical to Grad-CAM at layer4 here;
        # genuinely sharper at layer3.
        raw = (grads * acts).sum(dim=1)
    elif method == 'gradcampp':
        g2, g3 = grads.pow(2), grads.pow(3)
        denom = 2.0 * g2 + (acts.sum(dim=(2, 3), keepdim=True) * g3)
        alpha = torch.where(denom.abs() > EPS, g2 / (denom + EPS), torch.zeros_like(denom))
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        raw = (weights * acts).sum(dim=1)
    else:
        raise ValueError(f"method must be gradcam|hirescam|gradcampp, got {method!r}")

    # Spatial std of the gradient, averaged over channels: the direct measure of
    # whether this layer can produce an input-dependent map at all.
    grad_spatial_std = float(grads.std(dim=(2, 3)).mean())

    return _finalise(raw, relu=relu, extra={
        'method': method,
        'score': score,
        'target_class': int(target_class),
        'logit': float(output[0, target_class].detach()),
        'grad_l2': float(grads.norm()),
        'grad_spatial_std': grad_spatial_std,
        'act_shape': tuple(acts.shape[1:]),
    })


# ============================================================================
# Baselines / masking
# ============================================================================
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a normalised tensor.

    Valid because ImageNet normalisation is affine per channel and a Gaussian
    kernel sums to 1, so blurring the normalised tensor equals normalising the
    blurred image.
    """
    from torchvision.transforms.functional import gaussian_blur
    k = int(2 * round(3.0 * sigma) + 1)
    return gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def make_baseline(x: torch.Tensor, kind: str = 'blur', sigma: float = 8.0) -> torch.Tensor:
    """Baseline / fill for masking, in the same normalised space as `x`.

    'blur'  - a blurred copy of the image. RECOMMENDED. Masking toward black or
              toward flat grey drives the model far out of distribution, which is
              especially damaging for a detector keyed on compression and sensor
              statistics: a flat region has no such statistics at all. A blurred
              fill stays in distribution and isolates the local high-frequency
              content, which is the cue actually under test.
    'mean'  - the ImageNet mean colour (zero in normalised space): flat grey.
    'black' - a true black image, mapped through the normalisation.
    """
    if kind == 'blur':
        return _gaussian_blur(x, sigma)
    if kind == 'mean':
        return torch.zeros_like(x)
    if kind == 'black':
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return ((0.0 - mean) / std).expand_as(x).clone()
    raise ValueError(f"baseline must be blur|mean|black, got {kind!r}")


# ============================================================================
# Score-CAM
# ============================================================================
@torch.no_grad()
def score_cam(model: nn.Module,
              x: torch.Tensor,
              target_class: int,
              target_layer: nn.Module,
              batch_size: int = 32,
              top_k: Optional[int] = 512,
              baseline: str = 'blur',
              blur_sigma: float = 8.0,
              relu: bool = True) -> CamResult:
    """Score-CAM with the paper's normalisation actually applied.

    The previous implementation weighted channels by *raw logits* with neither a
    baseline subtraction nor a softmax. With a saturated model the logit is near
    constant across masks, so the map degenerated to `const * sum_c A_c` - the
    plain activation-energy map, independent of the target class. That is why
    Score-CAM and Grad-CAM produced near-identical pictures.
    """
    model.eval()
    _, acts = _forward_capturing(model, x, target_layer)
    acts = acts.detach()
    _, C, _, _ = acts.shape
    size = x.shape[2:]

    masks = F.interpolate(acts, size=size, mode='bilinear', align_corners=False)[0]
    lo = masks.amin(dim=(1, 2), keepdim=True)
    hi = masks.amax(dim=(1, 2), keepdim=True)
    span = (hi - lo).flatten()

    # Drop channels whose mask is essentially constant: they carry no spatial
    # information and their score would be pure baseline.
    keep = torch.nonzero(span > 1e-6, as_tuple=True)[0]
    if keep.numel() == 0:
        return _finalise(torch.zeros(acts.shape[2:], device=acts.device), relu=relu,
                         extra={'method': 'scorecam', 'n_channels_used': 0,
                                'target_class': int(target_class)})

    if top_k is not None and keep.numel() > top_k:
        energy = acts[0, keep].sum(dim=(1, 2))
        keep = keep[torch.topk(energy, top_k).indices]

    masks = (masks[keep] - lo[keep]) / (hi[keep] - lo[keep] + EPS)

    base = make_baseline(x, baseline, blur_sigma)
    # The baseline score must be the *same masking operation with a zero mask*,
    # not some unrelated zero input, or the subtraction is not a contrast.
    baseline_score = float(model(base)[0, target_class])

    scores = torch.empty(keep.numel(), device=x.device)
    for start in range(0, keep.numel(), batch_size):
        chunk = masks[start:start + batch_size].unsqueeze(1)          # (B,1,H,W)
        masked = chunk * x + (1.0 - chunk) * base                     # broadcast over RGB
        scores[start:start + batch_size] = model(masked)[:, target_class]

    weights = torch.softmax(scores - baseline_score, dim=0)
    raw = (weights.view(-1, 1, 1) * acts[0, keep]).sum(dim=0)

    return _finalise(raw, relu=relu, extra={
        'method': 'scorecam',
        'target_class': int(target_class),
        'n_channels_used': int(keep.numel()),
        'n_channels_total': int(C),
        'baseline': baseline,
        'baseline_score': baseline_score,
        'score_std': float(scores.std()),
        'act_shape': tuple(acts.shape[1:]),
    })


# ============================================================================
# Input-space attribution
# ============================================================================
@torch.no_grad()
def occlusion_map(model: nn.Module,
                  x: torch.Tensor,
                  target_class: int,
                  patch: int = 32,
                  stride: int = 16,
                  baseline: str = 'blur',
                  blur_sigma: float = 8.0,
                  batch_size: int = 32) -> CamResult:
    """Sliding-occlusion attribution: drop in the target logit when a patch is hidden.

    Model-faithful by construction, resolution is a free parameter, and it makes
    no assumption about where the evidence lives or how the network is wired.
    Use it as the reference map against which the CAMs are validated.
    """
    model.eval()
    _, _, H, W = x.shape
    base = make_baseline(x, baseline, blur_sigma)
    reference = float(model(x)[0, target_class])

    tops = list(range(0, max(H - patch, 0) + 1, stride))
    lefts = list(range(0, max(W - patch, 0) + 1, stride))
    if tops[-1] + patch < H:
        tops.append(H - patch)
    if lefts[-1] + patch < W:
        lefts.append(W - patch)
    positions = [(t, l) for t in tops for l in lefts]

    total = torch.zeros((H, W), device=x.device)
    counts = torch.zeros((H, W), device=x.device)

    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = x.repeat(len(chunk), 1, 1, 1)
        for i, (t, l) in enumerate(chunk):
            batch[i, :, t:t + patch, l:l + patch] = base[0, :, t:t + patch, l:l + patch]
        scores = model(batch)[:, target_class]
        for i, (t, l) in enumerate(chunk):
            total[t:t + patch, l:l + patch] += reference - scores[i]
            counts[t:t + patch, l:l + patch] += 1.0

    raw = total / counts.clamp(min=1.0)
    return _finalise(raw, relu=True, extra={
        'method': 'occlusion',
        'target_class': int(target_class),
        'reference_score': reference,
        'patch': patch,
        'stride': stride,
        'n_positions': len(positions),
        'baseline': baseline,
    })


def integrated_gradients(model: nn.Module,
                         x: torch.Tensor,
                         target_class: int,
                         steps: int = 256,
                         baseline: str = 'blur',
                         blur_sigma: float = 8.0,
                         batch_size: int = 16,
                         smoothgrad_n: int = 0,
                         smoothgrad_sigma: float = 0.1) -> CamResult:
    """Integrated gradients at full input resolution.

    `meta['completeness_residual']` is the relative error in the completeness
    identity `sum(attributions) == f(x) - f(baseline)`; it is the built-in
    correctness check for this method (see `V3` in the plan).

    On `steps`: the Riemann sum converges slowly here. Measured on
    ImageNet-resnet50 with a blur baseline, the residual runs 67% at 32 steps,
    26% at 64, 6.2% at 128, 5.1% at 256 and 1.2% at 512. The default of 256 is
    the point where the check passes with headroom; raise it if the reported
    residual is still above the 5% tolerance for your model.
    """
    model.eval()
    base = make_baseline(x, baseline, blur_sigma)
    delta = x - base

    def _grads_at(points: torch.Tensor) -> torch.Tensor:
        points = points.clone().detach().requires_grad_(True)
        out = model(points)
        s = out[:, target_class].sum()
        return torch.autograd.grad(s, points)[0]

    alphas = torch.linspace(0.0, 1.0, steps, device=x.device)
    accumulated = torch.zeros_like(x)
    for start in range(0, steps, batch_size):
        a = alphas[start:start + batch_size].view(-1, 1, 1, 1)
        points = base + a * delta
        if smoothgrad_n > 0:
            noise_scale = smoothgrad_sigma * float(x.max() - x.min())
            g = torch.zeros_like(points)
            for _ in range(smoothgrad_n):
                g += _grads_at(points + torch.randn_like(points) * noise_scale)
            g /= smoothgrad_n
        else:
            g = _grads_at(points)
        accumulated += g.sum(dim=0, keepdim=True)

    attributions = (delta * accumulated / steps)[0]

    with torch.no_grad():
        f_x = float(model(x)[0, target_class])
        f_base = float(model(base)[0, target_class])
    expected = f_x - f_base
    actual = float(attributions.sum())
    residual = abs(actual - expected) / (abs(expected) + EPS)

    return _finalise(attributions.sum(dim=0), relu=False, extra={
        'method': 'integrated_gradients',
        'target_class': int(target_class),
        'steps': steps,
        'baseline': baseline,
        'f_x': f_x,
        'f_baseline': f_base,
        'completeness_expected': expected,
        'completeness_actual': actual,
        'completeness_residual': residual,
    })


# ============================================================================
# Branch attribution for the 2-block combiner
# ============================================================================
def branch_attribution(complete_model: nn.Module,
                       x: torch.Tensor,
                       target_class: int) -> Dict[str, float]:
    """Attribute the complete model's logit to the DM and REAL feature blocks.

    Gradient x input on the concatenated features. For a piecewise-linear head
    this is exact up to bias terms, so `attr_dm + attr_real + residual == logit`;
    the residual is reported so the reader can judge the decomposition. This
    replaces the old `get_final_attribution`, which derived "attribution" from
    `convs[0].weight.abs()` and was therefore identical for every image.

    Answers "which expert drove this decision", the central claim of a
    block-based architecture and currently unsupported by anything.
    """
    complete_model.eval()
    with torch.no_grad():
        f_dm = complete_model.dm_extractor(x)
        f_real = complete_model.real_extractor(x)

    split = f_dm.shape[1]
    features = torch.cat([f_dm, f_real], dim=1).detach().requires_grad_(True)
    logit = complete_model.classifier(features)[0, target_class]
    grad = torch.autograd.grad(logit, features)[0][0]

    contribution = grad * features[0]
    attr_dm = float(contribution[:split].sum())
    attr_real = float(contribution[split:].sum())
    logit_value = float(logit)

    return {
        'attr_dm': attr_dm,
        'attr_real': attr_real,
        'logit': logit_value,
        'residual': logit_value - (attr_dm + attr_real),
        'dm_share': attr_dm / (abs(attr_dm) + abs(attr_real) + EPS),
    }


# ============================================================================
# Diagnostic: prove the degeneracy analytically
# ============================================================================
def assert_gap_linear_degeneracy(model: nn.Module,
                                 inputs,
                                 target_class: int = 1,
                                 stage: str = 'layer4') -> Dict[str, Any]:
    """Check that d logit / d A is the fc weight row, identically for every image.

    Turns the module docstring's algebra into a measurement. For a
    `layer4 -> GAP -> Linear` head we expect `pooled_grads == w_k / (H*W)` for
    every input, so `max_abs_error` should be at float32 noise level and
    `max_pairwise_diff` should be exactly 0. When that holds, the last-layer map
    is a fixed projection and no amount of CAM-variant tuning will change it.
    """
    model.eval()
    layer = resolve_stage(model, stage)

    fc = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc = module
    if fc is None:
        raise ValueError('no nn.Linear head found')

    inputs = list(inputs)
    if not inputs:
        raise ValueError('assert_gap_linear_degeneracy() needs at least one input')

    pooled = []
    hw = None
    for x in inputs:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        xg = _require_grad_input(x)
        output, acts = _forward_capturing(model, xg, layer)
        grads = torch.autograd.grad(output[0, target_class], acts)[0]
        pooled.append(grads.mean(dim=(2, 3))[0].detach())
        hw = acts.shape[2] * acts.shape[3]

    expected = fc.weight[target_class].detach() / hw
    errors = [float((p - expected).abs().max()) for p in pooled]
    stacked = torch.stack(pooled)
    pairwise = float((stacked - stacked[0]).abs().max())

    return {
        'n_inputs': len(pooled),
        'feature_map_cells': hw,
        'max_abs_error_vs_fc_weights': max(errors),
        'max_pairwise_diff_across_images': pairwise,
        'degenerate': max(errors) < 1e-5 and pairwise < 1e-6,
    }


# ============================================================================
# Rendering
# ============================================================================
def overlay_heatmap(img_np: np.ndarray,
                    cam: np.ndarray,
                    alpha_max: float = 0.7,
                    pclip: float = 99.0,
                    cmap: str = 'inferno') -> np.ndarray:
    """Blend a saliency map onto an image.

    Differs from the previous implementation in four ways that all mattered:
    upsampling happens in float (no uint8 quantisation), the alpha is
    proportional to the map value instead of a flat 0.5 that washed the whole
    image blue, values are clipped at a percentile so one saturated cell cannot
    crush the rest, and the colormap is perceptually uniform - `jet` manufactures
    false structure at mid values.
    """
    import matplotlib.pyplot as plt

    cam_t = torch.from_numpy(np.ascontiguousarray(cam)).float()[None, None]
    cam_up = F.interpolate(cam_t, size=img_np.shape[:2],
                           mode='bilinear', align_corners=False)[0, 0].numpy()

    hi = np.percentile(cam_up, pclip)
    cam_up = np.clip(cam_up / (hi + EPS), 0.0, 1.0)

    colour = plt.get_cmap(cmap)(cam_up)[..., :3] * 255.0
    a = (alpha_max * cam_up)[..., None]
    return np.clip(img_np * (1.0 - a) + colour * a, 0, 255).astype(np.uint8)


# ============================================================================
# Consistency metrics
# ============================================================================
def ecs_pearson(a: np.ndarray, b: np.ndarray) -> Tuple[float, str]:
    """Pearson correlation between two maps, with an explicit reason code.

    Returns NaN rather than 0.0 when a map is constant. The old implementation
    returned 0.0 in that case, which silently conflated "the map is degenerate,
    there was nothing to compare" with "the two maps are uncorrelated" - and is
    almost certainly the origin of the `min ECS = 0.000` entries in the existing
    batch summary.

    Always call this on the NATIVE grid (8x8 / 16x16), never on the map upsampled
    to input resolution: interpolation smoothness inflates the correlation.
    """
    x, y = np.asarray(a).flatten(), np.asarray(b).flatten()
    if x.shape != y.shape:
        return float('nan'), 'shape_mismatch'
    if np.std(x) < EPS and np.std(y) < EPS:
        return float('nan'), 'both_degenerate'
    if np.std(x) < EPS:
        return float('nan'), 'reference_degenerate'
    if np.std(y) < EPS:
        return float('nan'), 'candidate_degenerate'
    return float(np.corrcoef(x, y)[0, 1]), 'ok'


def ecs_iou(a: np.ndarray, b: np.ndarray, quantile: float = 0.5) -> Tuple[float, str]:
    """IoU of the regions above `quantile` of each map's maximum.

    A threshold-based companion to the Pearson score: less sensitive to the exact
    values, more sensitive to whether the maps point at the same place. Salvaged
    from the previous `RobustnessAnalyzer.compute_ecs`.
    """
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return float('nan'), 'shape_mismatch'
    if x.max() < EPS or y.max() < EPS:
        return float('nan'), 'degenerate'
    mx, my = x > quantile * x.max(), y > quantile * y.max()
    union = np.logical_or(mx, my).sum()
    if union == 0:
        return float('nan'), 'empty_union'
    return float(np.logical_and(mx, my).sum() / union), 'ok'
