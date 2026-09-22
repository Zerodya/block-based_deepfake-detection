#!/usr/bin/env python3
"""Decide whether the saliency maps are explanations or a fixed spatial prior.

Motivation. The existing batch summary reports a robustness score (Pearson
correlation between a map and the same map recomputed after a destructive
transform) of ~0.986 with a minimum of 0.875 for the REAL branch - and ~0.972
even on generated images, i.e. essentially unchanged when the input class
changes. A map that genuinely tracks image content cannot survive a 0.25x
downscale and JPEG QF25 that intact.

The explanation is analytic, not a coding slip: for a `layer4 -> GAP -> Linear`
head, d(logit)/d(activation) is constant in space and independent of the image,
so a last-layer CAM is a FIXED linear projection of the activation tensor (see
src/dfx/cam.py). This script measures that claim instead of asserting it, and
quantifies how much of each map is prior versus signal.

Tests, in decreasing evidence-per-second:

  T1  analytic    - is d(logit)/d(A) literally the fc weight row, for every image?
  T2  flip        - does the map flip when the image flips?
  T3  prior       - mean map over N images; how much variance does it explain?
  T4  content-free- maps on grey / noise / patch-shuffled / unrelated inputs
  T5  faithfulness- deletion AUC vs random ordering and vs the mean-prior map
  T6  randomize   - cascading model randomisation (Adebayo et al., NeurIPS 2018)
  T7  null control- the same consistency sweep on UNTRAINED weights

Usage:
  python diagnose_prior.py --models_dir M --backbone resnet50 \
      --dataset_dir D --n 200 --stages layer3,layer4
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, HERE)

from dfx.dataset_classes import get_trans
from dfx.cam import (
    compute_cam, resolve_stage, make_baseline,
    assert_gap_linear_degeneracy, ecs_pearson,
)
from dfx.postprocessing import (
    jpeg_compression, gaussian_blur, resize_down_up, brightness_adjust,
)
from single_test import (
    SAVED_NAME_MAP, CLASS_NAMES, load_complete_model, load_base_model,
    CompleteModel2Blocks,
)


IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models_dir', required=True)
    p.add_argument('--backbone', required=True, choices=sorted(SAVED_NAME_MAP))
    p.add_argument('--dataset_dir', required=True,
                   help='Directory of images (searched recursively)')
    p.add_argument('--n', type=int, default=200, help='Images to sample')
    p.add_argument('--stages', default='layer3,layer4')
    p.add_argument('--method', default='gradcam',
                   choices=['gradcam', 'hirescam', 'gradcampp'])
    p.add_argument('--output_dir', default='../explanation_results/_diagnostics')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--deletion_steps', type=int, default=20)
    p.add_argument('--skip', default='', help='Comma-separated test ids to skip')
    return p.parse_args()


def collect(dataset_dir, n, seed):
    paths = [os.path.join(r, f)
             for r, _, fs in os.walk(dataset_dir) for f in fs
             if os.path.splitext(f)[1].lower() in IMAGE_EXT]
    if not paths:
        raise SystemExit(f'no images under {dataset_dir}')
    random.Random(seed).shuffle(paths)
    return sorted(paths[:n])


# ============================================================================
# T3 - the prior, and how much of each map it accounts for
# ============================================================================
def test_prior(maps):
    """Decompose a stack of maps into a shared mean and per-image residuals.

    `explained_variance` near 1 means the mean map IS the map: every image gets
    the same picture. The residual is the part that actually depends on the
    image, and recomputing consistency on residuals is the honest robustness
    number - the raw score is inflated by the shared component.
    """
    stack = np.stack(maps)
    mean_map = stack.mean(axis=0)

    to_mean = [ecs_pearson(m, mean_map)[0] for m in stack]
    pairwise = []
    rng = random.Random(0)
    for _ in range(min(500, len(stack) * (len(stack) - 1) // 2)):
        i, j = rng.sample(range(len(stack)), 2)
        pairwise.append(ecs_pearson(stack[i], stack[j])[0])

    residual = stack - mean_map
    return {
        'n': len(stack),
        'mean_corr_to_mean_map': float(np.nanmean(to_mean)),
        'mean_pairwise_corr': float(np.nanmean(pairwise)),
        'explained_variance_by_mean': float(
            mean_map.var() / (stack.var(axis=0).mean() + mean_map.var() + 1e-12)),
        'residual_std_over_map_std': float(residual.std() / (stack.std() + 1e-12)),
        'mean_map': mean_map,
    }


# ============================================================================
# T4 - content-free and content-scrambled inputs
# ============================================================================
def patch_shuffle(x, block=8, seed=0):
    """Destroy global layout, preserve local statistics.

    The sharpest control here: a genuine texture/artifact detector should keep a
    similar distribution of values but produce a DIFFERENT map; a positional
    prior produces the same map either way.
    """
    _, _, H, W = x.shape
    nh, nw = H // block, W // block
    patches = x[:, :, :nh * block, :nw * block]
    patches = patches.unfold(2, block, block).unfold(3, block, block)
    patches = patches.contiguous().view(1, x.shape[1], nh * nw, block, block)
    perm = torch.randperm(nh * nw, generator=torch.Generator().manual_seed(seed))
    patches = patches[:, :, perm]
    patches = patches.view(1, x.shape[1], nh, nw, block, block)
    out = patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(1, x.shape[1],
                                                              nh * block, nw * block)
    result = x.clone()
    result[:, :, :nh * block, :nw * block] = out
    return result


def test_content_free(net, layer, x, other_x, target, method, seed):
    ref = compute_cam(net, x, target, layer, method=method).map
    probes = {
        'grey': torch.zeros_like(x),
        'noise': torch.randn_like(x),
        'patch_shuffled': patch_shuffle(x, 8, seed),
        'unrelated_image': other_x,
    }
    out = {}
    for name, probe in probes.items():
        m = compute_cam(net, probe, target, layer, method=method).map
        r, reason = ecs_pearson(ref, m)
        out[name] = {'r': r, 'reason': reason}
    return out


# ============================================================================
# T5 - faithfulness
# ============================================================================
@torch.no_grad()
def deletion_curve(net, x, target, saliency, steps=20, baseline='blur'):
    """Replace the most-salient pixels first; a faithful map drops the logit fast."""
    H, W = x.shape[2:]
    base = make_baseline(x, baseline)
    sal = torch.from_numpy(np.ascontiguousarray(saliency)).float()[None, None]
    sal = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)
    order = torch.argsort(sal.flatten(), descending=True).to(x.device)

    scores = []
    for i in range(steps + 1):
        keep = int(order.numel() * i / steps)
        mask = torch.zeros(order.numel(), device=x.device)
        if keep:
            mask[order[:keep]] = 1.0
        mask = mask.view(1, 1, H, W)
        scores.append(float(net((1 - mask) * x + mask * base)[0, target]))
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return scores, float(integrate(scores, dx=1.0 / steps))


def test_faithfulness(net, x, target, cam_map, prior_map, steps, seed):
    rng = np.random.default_rng(seed)
    random_map = rng.random(cam_map.shape)
    curves = {}
    for name, m in (('cam', cam_map), ('prior', prior_map), ('random', random_map)):
        scores, auc = deletion_curve(net, x, target, m, steps)
        curves[name] = {'scores': scores, 'auc': auc}
    # A CAM that does not beat the shared prior is, by definition, no better
    # than a constant spatial prior. Lower deletion AUC is better.
    curves['cam_beats_prior'] = curves['cam']['auc'] < curves['prior']['auc']
    curves['cam_beats_random'] = curves['cam']['auc'] < curves['random']['auc']
    return curves


# ============================================================================
# T7 - untrained-model null control for the consistency score
# ============================================================================
def test_null_ecs(make_untrained, pil_images, transform, device, stage, method,
                  sweep):
    """Run the whole consistency sweep on an UNTRAINED network.

    The point: if a randomly initialised model scores as highly as the trained
    one, the consistency score measures nothing about what the model learned.
    Without this control a high score reads as evidence that the explanation is
    stable; with it, the number is exposed as a property of the architecture.
    Report it beside every consistency figure in the thesis.

    The pipeline here MUST mirror `single_test.run_robustness` exactly - the
    post-processing is applied to the source image at its native resolution and
    only then resized by the training transform. Applying it to an
    already-downscaled tensor is far more destructive and yields a number that
    is not comparable with the one being controlled for.
    """
    net = make_untrained()
    layer = resolve_stage(net.dm_extractor, stage)
    scores = []
    for img in pil_images:
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            k = int(net(x).argmax())
        ref = compute_cam(net, x, k, layer, method=method).map
        for _, fn in sweep:
            xt = transform(fn(img)).unsqueeze(0).to(device)
            cand = compute_cam(net, xt, k, layer, method=method).map
            r, reason = ecs_pearson(ref, cand)
            if reason == 'ok':
                scores.append(r)
    return {'mean_ecs_untrained': float(np.nanmean(scores)) if scores else float('nan'),
            'min_ecs_untrained': float(np.nanmin(scores)) if scores else float('nan'),
            'n': len(scores)}


# ============================================================================
# T6 - cascading model randomisation (Adebayo et al., NeurIPS 2018)
# ============================================================================
def _reinit(module):
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def test_randomization(net, owner, x, target, stage, method):
    """If the map survives randomising the weights, it is not an explanation.

    Randomisation is CUMULATIVE, top-down: head, then head+layer4, then
    head+layer4+layer3. Each step recomputes the map and correlates it against
    the trained model's map.
    """
    import copy

    reference = compute_cam(net, x, target, resolve_stage(owner, stage),
                            method=method).map

    net_c = copy.deepcopy(net)
    owner_c = _matching_owner(net_c, net, owner)

    def _heads(module):
        return [m for m in module.modules() if isinstance(m, nn.Linear)]

    cascade = [('head', _heads(net_c)),
               ('layer4', [resolve_stage(owner_c, 'layer4')]),
               ('layer3', [resolve_stage(owner_c, 'layer3')])]

    out = {}
    for label, modules in cascade:
        for m in modules:
            _reinit(m)
        candidate = compute_cam(net_c, x, target, resolve_stage(owner_c, stage),
                                method=method).map
        r, reason = ecs_pearson(reference, candidate)
        out[f'after_randomizing_{label}'] = {'r': r, 'reason': reason}
    return out


def _matching_owner(net_copy, net_original, owner):
    """Locate, in a deepcopy of the model, the branch that `owner` names."""
    for name, child in net_original.named_children():
        if child is owner:
            return getattr(net_copy, name)
    return net_copy


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'Device: {device}')

    transform = get_trans(model_name=args.backbone)
    input_size = int(transform(Image.new('RGB', (512, 512))).shape[-1])

    model = load_complete_model(args.backbone, args.models_dir, device, input_size)
    branches = [('DM-branch', model.dm_extractor), ('REAL-branch', model.real_extractor)]

    paths = collect(args.dataset_dir, args.n, args.seed)
    print(f'Sampled {len(paths)} images from {args.dataset_dir}\n')

    def load(path):
        return transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)

    report = {'n_images': len(paths), 'stages': stages, 'method': args.method,
              'input_size': input_size}

    # ---- T1 -----------------------------------------------------------------
    if 'T1' not in skip:
        print('[T1] Analytic degeneracy of the last-layer gradient')
        print('     Expect: gradient == fc weight row / HW, identical for all images.')
        saved = SAVED_NAME_MAP.get(args.backbone, args.backbone)
        base = load_base_model(args.backbone, os.path.join(
            args.models_dir, 'real', f'{saved}.pt'), device)
        res = assert_gap_linear_degeneracy(
            base, [load(p) for p in paths[:20]], target_class=1, stage='layer4')
        report['T1_analytic'] = res
        print(f"     max |grad - w_k/HW|          = {res['max_abs_error_vs_fc_weights']:.3e}")
        print(f"     max difference across images = {res['max_pairwise_diff_across_images']:.3e}")
        print(f"     => last-layer CAM is a FIXED projection: {res['degenerate']}\n")

    # ---- collect maps once, reuse for T2/T3/T5 ------------------------------
    print('[..] Computing maps')
    maps = {(stage, label): [] for stage in stages for label, _ in branches}
    flips = {(stage, label): [] for stage in stages for label, _ in branches}
    preds = []
    for i, path in enumerate(paths, 1):
        x = load(path)
        with torch.no_grad():
            k = int(model(x).argmax())
        preds.append(k)
        for stage in stages:
            for label, owner in branches:
                layer = resolve_stage(owner, stage)
                m = compute_cam(model, x, k, layer, method=args.method).map
                maps[(stage, label)].append(m)
                if 'T2' not in skip:
                    mf = compute_cam(model, torch.flip(x, dims=[3]), k, layer,
                                     method=args.method).map
                    flips[(stage, label)].append(ecs_pearson(np.fliplr(mf), m)[0])
        if i % 25 == 0 or i == len(paths):
            print(f'     {i}/{len(paths)}')
    print()

    # ---- T2 -----------------------------------------------------------------
    if 'T2' not in skip:
        print('[T2] Flip equivariance, averaged over all sampled images')
        print('     Interpret comparatively: layer3 should clearly exceed layer4,')
        print('     and both should exceed the score of the shared prior map.')
        report['T2_flip'] = {}
        for key, vals in flips.items():
            stage, label = key
            mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
            report['T2_flip'][f'{stage}|{label}'] = {'mean_r': mean, 'std_r': std}
            print(f'     {stage:>7} {label:<12} r = {mean:+.3f} +/- {std:.3f}')
        print()

    # ---- T3 -----------------------------------------------------------------
    print('[T3] Shared spatial prior')
    report['T3_prior'] = {}
    priors = {}
    for key, stack in maps.items():
        stage, label = key
        res = test_prior(stack)
        priors[key] = res.pop('mean_map')
        report['T3_prior'][f'{stage}|{label}'] = res
        print(f"     {stage:>7} {label:<12} corr-to-mean={res['mean_corr_to_mean_map']:+.3f}  "
              f"pairwise={res['mean_pairwise_corr']:+.3f}  "
              f"var explained by mean={res['explained_variance_by_mean']:.3f}")
    _plot_priors(priors, os.path.join(args.output_dir, 'mean_prior_maps.png'))
    print()

    # ---- T4 -----------------------------------------------------------------
    if 'T4' not in skip and len(paths) >= 2:
        print('[T4] Content-free inputs   (high r = the map ignores the image)')
        report['T4_content_free'] = {}
        x0, x1 = load(paths[0]), load(paths[1])
        for stage in stages:
            for label, owner in branches:
                res = test_content_free(model, resolve_stage(owner, stage), x0, x1,
                                        preds[0], args.method, args.seed)
                report['T4_content_free'][f'{stage}|{label}'] = res
                summary = '  '.join(f"{k}={v['r']:+.2f}" for k, v in res.items())
                print(f'     {stage:>7} {label:<12} {summary}')
        print()

    # ---- T5 -----------------------------------------------------------------
    if 'T5' not in skip:
        print('[T5] Faithfulness: deletion AUC (lower is better)')
        report['T5_faithfulness'] = {}
        x0 = load(paths[0])
        for stage in stages:
            for label, owner in branches:
                res = test_faithfulness(model, x0, preds[0],
                                        maps[(stage, label)][0], priors[(stage, label)],
                                        args.deletion_steps, args.seed)
                report['T5_faithfulness'][f'{stage}|{label}'] = res
                print(f"     {stage:>7} {label:<12} cam={res['cam']['auc']:+.3f}  "
                      f"prior={res['prior']['auc']:+.3f}  random={res['random']['auc']:+.3f}  "
                      f"beats prior: {res['cam_beats_prior']}")
        print()

    # ---- T6 -----------------------------------------------------------------
    if 'T6' not in skip:
        print('[T6] Cascading model randomisation (Adebayo et al. 2018)')
        print('     A map that survives randomisation is not an explanation.')
        report['T6_randomization'] = {}
        x0 = load(paths[0])
        for stage in stages:
            for label, owner in branches:
                try:
                    res = test_randomization(model, owner, x0, preds[0], stage, args.method)
                except Exception as exc:
                    res = {'error': str(exc)}
                report['T6_randomization'][f'{stage}|{label}'] = res
                print(f'     {stage:>7} {label:<12} ' + '  '.join(
                    f"{k.replace('after_randomizing_', '')}={v['r']:+.2f}"
                    for k, v in res.items() if isinstance(v, dict)))
        print()

    # ---- T7 -----------------------------------------------------------------
    if 'T7' not in skip:
        print('[T7] Untrained-model null control for the consistency score')
        print('     If random weights score as high as the trained model, the')
        print('     consistency score measures nothing the model learned.')
        sweep = [('JPEG QF25', jpeg_compression(25)),
                 ('Blur r=2.0', gaussian_blur(2.0)),
                 ('Resize 0.25x', resize_down_up(0.25)),
                 ('Brightness -20%', brightness_adjust(0.8))]
        pil_images = [Image.open(q).convert('RGB') for q in paths[:10]]

        def make_untrained():
            torch.manual_seed(args.seed)
            net = CompleteModel2Blocks(args.backbone, args.models_dir, device,
                                       input_size)
            for m in net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            return net.eval()

        report['T7_null_control'] = {}
        for stage in stages:
            res = test_null_ecs(make_untrained, pil_images, transform, device,
                                stage, args.method, sweep)
            report['T7_null_control'][stage] = res
            print(f"     {stage:>7} untrained mean ECS = "
                  f"{res['mean_ecs_untrained']:+.3f} "
                  f"(min {res['min_ecs_untrained']:+.3f}, n={res['n']})")
        print()

    path = os.path.join(args.output_dir, 'diagnostics.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'Report: {path}')
    _print_verdict(report)


def _plot_priors(priors, path):
    fig, axes = plt.subplots(1, len(priors), figsize=(3.6 * len(priors), 3.6), squeeze=False)
    for ax, (key, m) in zip(axes[0], priors.items()):
        im = ax.imshow(m, cmap='inferno')
        ax.set_title(f'{key[0]} · {key[1]}\nmean map over all images', fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('The shared spatial prior, made visible', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'     Saved: {path}')


def _print_verdict(report):
    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)
    t1 = report.get('T1_analytic', {})
    if t1:
        print(f"  Last-layer gradient is input-independent : {t1.get('degenerate')}")
    for key, val in (report.get('T2_flip') or {}).items():
        verdict = ('follows content' if val['mean_r'] > 0.5
                   else 'weak' if val['mean_r'] > 0.2 else 'FIXED PRIOR')
        print(f"  flip r  {key:<24} {val['mean_r']:+.3f}   {verdict}")
    for key, val in (report.get('T3_prior') or {}).items():
        print(f"  prior   {key:<24} corr-to-mean {val['mean_corr_to_mean_map']:+.3f}")
    for key, val in (report.get('T7_null_control') or {}).items():
        print(f"  UNTRAINED consistency at {key:<15} "
              f"{val['mean_ecs_untrained']:+.3f}   <- compare with the trained figure")
    print('=' * 70)


if __name__ == '__main__':
    main()
