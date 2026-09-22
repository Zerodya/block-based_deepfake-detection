#!/usr/bin/env python3
"""DeepFeatureX explainability - 2 blocks (DM + REAL).

Produces attribution maps for a single image, plus the diagnostics needed to
decide whether those maps mean anything.

Read `src/dfx/cam.py`'s module docstring first. The short version: for a
`layer4 -> GAP -> Linear` head the gradient of the logit w.r.t. the layer4
activations is *constant in space and independent of the image*, so a last-layer
CAM is a fixed linear projection of the activation tensor. Last-layer maps are
therefore expected to look alike across images; that is a property of the
architecture, not a bug in this script. `--stages layer3,layer4` and the
input-space methods are where genuinely input-dependent attribution lives.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dfx.architecture import backbone
from dfx.dataset_classes import get_trans
from dfx.cam import (
    compute_cam,
    score_cam,
    occlusion_map,
    integrated_gradients,
    branch_attribution,
    resolve_stage,
    disable_inplace_relu,
    overlay_heatmap,
    ecs_pearson,
    ecs_iou,
)
from dfx.postprocessing import (
    jpeg_compression,
    gaussian_blur,
    resize_down_up,
    screenshot_simulation,
    brightness_adjust,
    contrast_adjust,
    sharpness_adjust,
    color_adjust,
)

CLASS_NAMES = {0: 'DM', 1: 'REAL'}


def class_label(k):
    """Name a target class. In --sanity mode k is an ImageNet class id."""
    return CLASS_NAMES.get(k, f'class{k}')

SAVED_NAME_MAP = {
    'efficientnet_b0': 'effb0', 'efficientnet_b4': 'effb4',
    'efficientnet_widese_b0': 'effb0', 'efficientnet_widese_b4': 'effb4',
    'resnet18': 'res18', 'resnet34': 'res34', 'resnet50': 'res50',
    'resnet101': 'res101', 'resnet152': 'res152',
    'resnext101': 'resnext101',
    'densenet121': 'dense121', 'densenet161': 'dense161',
    'densenet169': 'dense169', 'densenet201': 'dense201',
    'vit_b_16': 'vitb16', 'vit_b_32': 'vitb32',
    'vit_l_16': 'vitl16', 'vit_l_32': 'vitl32',
}


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='DeepFeatureX Explainability - 2 Blocks (DM + REAL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # normal run
  python %(prog)s --models_dir /path/to/models --backbone resnet50 \\
      --image_path img.png

  # sanity gate: does the CAM machinery work at all, on a model with known
  # ground truth? Run this FIRST - nothing else is interpretable until it passes.
  python %(prog)s --sanity --image_path img.png --models_dir . --backbone resnet50

  # full attribution including the input-space methods (slower)
  python %(prog)s --models_dir M --backbone resnet50 --image_path img.png \\
      --occlusion --integrated_gradients
        """)

    p.add_argument('--models_dir', type=str, required=True,
                   help='Root dir containing dm_generated/, real/, complete/')
    p.add_argument('--backbone', type=str, required=True,
                   choices=sorted(SAVED_NAME_MAP))
    p.add_argument('--image_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='../explanation_results')

    p.add_argument('--mode', choices=['complete', 'per-block'], default='complete',
                   help="'complete' explains the prediction that is actually "
                        "displayed; 'per-block' explains each expert's own "
                        'hypothesis (the previous behaviour)')
    p.add_argument('--stages', type=str, default='layer3,layer4',
                   help='Comma-separated residual stages to attribute at. '
                        'layer4 is analytically degenerate for this head; '
                        'layer3 is where spatial gradient variation begins.')
    p.add_argument('--methods', type=str, default='gradcam,scorecam',
                   help='gradcam, hirescam, gradcampp, scorecam')
    p.add_argument('--score', choices=['logit', 'margin'], default='logit',
                   help="'margin' uses logit_target - logit_other, the real "
                        'decision function of a 2-class head')
    p.add_argument('--all_targets', action='store_true',
                   help='Explain both classes, not only the predicted one')

    p.add_argument('--scorecam_baseline', choices=['blur', 'mean', 'black'],
                   default='blur',
                   help='Masking fill. blur keeps the image in-distribution, '
                        'which matters for a detector keyed on local statistics.')
    p.add_argument('--scorecam_topk', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=32)

    p.add_argument('--occlusion', action='store_true',
                   help='Sliding-occlusion map: model-faithful reference')
    p.add_argument('--integrated_gradients', action='store_true',
                   help='Full-resolution IG with a completeness check')
    p.add_argument('--no_robustness', action='store_true',
                   help='Skip the 16-transform robustness sweep')
    p.add_argument('--robustness_stage', default='layer4',
                   help='Stage for the ECS sweep. Defaults to layer4 so the '
                        'numbers stay comparable with the historical batch '
                        'summary; layer3 is more informative.')
    p.add_argument('--seed', type=int, default=0,
                   help='Seed for the stochastic transforms, so ECS is reproducible')

    p.add_argument('--sanity', action='store_true',
                   help='Gate check: run on ImageNet-pretrained resnet50 with a '
                        'known class instead of the deepfake models')
    return p.parse_args()


def sanitize_filename(name):
    return re.sub(r'_+', '_', re.sub(r'[^\w\-.]', '_', name)).strip('_')


# ============================================================================
# Models
# ============================================================================
def load_base_model(backbone_name, model_path, device):
    model = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    disable_inplace_relu(model)
    return model.to(device).eval()


class FeatureExtractor(nn.Module):
    """Mirrors the wrapper used at training time (training-complete-model.py).

    NOTE: `children()[:-1]` keeps `avgpool`, so the output is (1, 2048, 1, 1)
    at any input resolution. That is precisely why the 224-vs-256 mismatch never
    raised an error and went unnoticed. It is also only structurally valid for
    the ResNet family - see resolve_stage() for the details.
    """

    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.features(x), 1)


class CompleteModel2Blocks(nn.Module):
    def __init__(self, backbone_name, models_dir, device, input_size=256):
        super().__init__()
        saved = SAVED_NAME_MAP.get(backbone_name, backbone_name)
        dm_path = os.path.join(models_dir, 'dm_generated', f'{saved}.pt')
        real_path = os.path.join(models_dir, 'real', f'{saved}.pt')

        for path, name in [(dm_path, 'DM'), (real_path, 'REAL')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f'{name} base model not found: {path}')
            print(f'    Found {name} base model: {path}')

        self.dm_extractor = FeatureExtractor(
            load_base_model(backbone_name, dm_path, device)).to(device)
        self.real_extractor = FeatureExtractor(
            load_base_model(backbone_name, real_path, device)).to(device)

        for p in list(self.dm_extractor.parameters()) + list(self.real_extractor.parameters()):
            p.requires_grad = False

        dummy = torch.randn(1, 3, input_size, input_size).to(device)
        with torch.no_grad():
            feat_dim = (self.dm_extractor(dummy).shape[1]
                        + self.real_extractor(dummy).shape[1])

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2)
        ).to(device)

    def forward(self, x):
        return self.classifier(torch.cat(
            [self.dm_extractor(x), self.real_extractor(x)], dim=1))


def load_complete_model(backbone_name, models_dir, device, input_size):
    saved = SAVED_NAME_MAP.get(backbone_name, backbone_name)
    complete_path = os.path.join(models_dir, 'complete', f'{saved}.pt')

    print('    Loading 2-block Complete Model...')
    model = CompleteModel2Blocks(backbone_name, models_dir, device, input_size)
    if not os.path.exists(complete_path):
        raise FileNotFoundError(
            f'Complete model not found: {complete_path}\n'
            'Train the complete model first, or check --models_dir (it must '
            'point INSIDE unbalancing-approach/).')
    model.load_state_dict(torch.load(complete_path, map_location=device, weights_only=False))
    print(f'    Loaded trained combiner: {complete_path}')
    return model.to(device).eval()


# ============================================================================
# Attribution driver
# ============================================================================
def branch_targets(model, mode):
    """Which (label, model, layer-owner) triples to attribute.

    In 'complete' mode both maps come from `model_complete`, so they explain the
    very logits printed in the title. In 'per-block' mode each expert explains
    its own hypothesis, which is a different - and weaker - claim.
    """
    if mode == 'complete':
        return [('DM-branch', model, model.dm_extractor),
                ('REAL-branch', model, model.real_extractor)]
    return [('DM-expert', model['dm'], model['dm']),
            ('REAL-expert', model['real'], model['real'])]


def run_attribution(owners, x, targets, stages, methods, args):
    """Compute every requested (stage, method, branch, target class) map."""
    results = {}
    for stage in stages:
        for label, net, owner in owners:
            layer = resolve_stage(owner, stage)
            for k in targets:
                for method in methods:
                    key = (stage, method, label, k)
                    if method == 'scorecam':
                        res = score_cam(net, x, k, layer,
                                        batch_size=args.batch_size,
                                        top_k=args.scorecam_topk,
                                        baseline=args.scorecam_baseline)
                    else:
                        res = compute_cam(net, x, k, layer,
                                          method=method, score=args.score)
                    results[key] = res
                    flag = ' [DEGENERATE]' if res.degenerate else ''
                    print(f'      {stage:>7} {method:<10} {label:<12} '
                          f'target={class_label(k):<6} '
                          f"grad_sd={res.meta.get('grad_spatial_std', float('nan')):.3e}{flag}")
    return results


def flip_equivariance(owners, x, target, stage, method, args):
    """V4: a content-following map flips with the image; a positional prior does not.

    Four forwards, no ambiguity. This is the single most decisive cheap test of
    whether a map is an explanation or a fixed spatial prior.
    """
    out = {}
    for label, net, owner in owners:
        layer = resolve_stage(owner, stage)
        a = compute_cam(net, x, target, layer, method=method, score=args.score)
        b = compute_cam(net, torch.flip(x, dims=[3]), target, layer,
                        method=method, score=args.score)
        r, reason = ecs_pearson(np.fliplr(b.map), a.map)
        out[label] = {'r': r, 'reason': reason}
        print(f'      {label:<12} r = {r:+.3f}  ({reason})')
    return out


# ============================================================================
# Rendering
# ============================================================================
def draw_panel(ax, img_np, result, title):
    """Render one map, or an explicit refusal when the map is degenerate."""
    if result.degenerate:
        ax.imshow(np.zeros_like(img_np) + 245)
        ax.add_patch(plt.Rectangle((0, 0), img_np.shape[1] - 1, img_np.shape[0] - 1,
                                   fill=False, hatch='///', edgecolor='#b00020', lw=1.5))
        ax.text(0.5, 0.5, 'no positive\nevidence', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='#b00020', fontweight='bold')
    else:
        ax.imshow(overlay_heatmap(img_np, result.map))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def figure_main(img_np, results, stages, methods, labels, target, out_path, suptitle):
    cols = 1 + len(methods) * len(labels)
    fig, axes = plt.subplots(len(stages), cols,
                             figsize=(3.3 * cols, 3.9 * len(stages)),
                             squeeze=False, layout='constrained')
    for r, stage in enumerate(stages):
        axes[r][0].imshow(img_np)
        axes[r][0].set_title(f'Original\n({stage})', fontsize=9)
        axes[r][0].axis('off')
        c = 1
        for method in methods:
            for label in labels:
                res = results[(stage, method, label, target)]
                grid = 'x'.join(str(v) for v in res.meta.get('grid_shape', ()))
                draw_panel(axes[r][c], img_np, res,
                           f'{method}: {label}\n{stage} ({grid})')
                c += 1
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'    Saved: {out_path}')


def figure_signed_and_scale(results, stages, methods, labels, target, out_path):
    """The signed map plus a colorbar on the TRUE value range.

    Min-max normalisation is what lets a map whose dynamic range is 1e-6 look
    like a confident explanation. This figure is the antidote: it shows both the
    suppressing evidence and the actual magnitudes.
    """
    items = [(s, m, l) for s in stages for m in methods for l in labels]
    fig, axes = plt.subplots(1, len(items), figsize=(4.0 * len(items), 4.4),
                             squeeze=False, layout='constrained')
    for ax, (stage, method, label) in zip(axes[0], items):
        res = results[(stage, method, label, target)]
        lim = float(np.abs(res.signed).max()) or 1.0
        im = ax.imshow(res.signed, cmap='coolwarm', vmin=-lim, vmax=lim)
        ax.set_title(f'{method}: {label}\n{stage}  range=[{res.meta["pre_relu_min"]:.2e}, '
                     f'{res.meta["pre_relu_max"]:.2e}]', fontsize=8)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Signed attribution on the true (unnormalised) value range',
                 fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'    Saved: {out_path}')


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    out_dir = os.path.join(args.output_dir,
                           sanitize_filename(os.path.splitext(
                               os.path.basename(args.image_path))[0]))
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    print(f'Output folder for this image: {out_dir}')

    # --- preprocessing: one source of truth, shared with training ----------
    # Previously this script hardcoded Resize((224,224)) while get_trans() - used
    # by every training and testing script - resizes to 256. The explainer was
    # running the models off their training distribution on a 7x7 grid instead
    # of 8x8, and nothing errored because FeatureExtractor keeps avgpool.
    transform = get_trans(model_name=args.backbone)
    input_size = int(transform(Image.new('RGB', (512, 512))).shape[-1])
    print(f'    Input size from get_trans(): {input_size}x{input_size}')

    img = Image.open(args.image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img.resize((input_size, input_size), Image.BILINEAR))

    # --- models -------------------------------------------------------------
    print('\n[1] Loading models...')
    if args.sanity:
        # V1 GATE. Known ground truth: if the map does not land on the object,
        # the bug is in this code. If it does, the machinery is correct and any
        # flatness on the deepfake models is a property of those models.
        import torchvision
        print('    SANITY MODE: ImageNet-pretrained resnet50')
        net = disable_inplace_relu(
            torchvision.models.resnet50(weights='DEFAULT')).to(device).eval()
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)
        target = int(probs.argmax())
        print(f'    ImageNet class {target}, p={probs[0, target]:.4f}')
        owners = [('imagenet', net, net)]
        targets = [target]
        names = {target: f'class{target}'}
        pred_class, prob_dm, prob_real = target, float('nan'), float('nan')
        attribution = None
    else:
        model = load_complete_model(args.backbone, args.models_dir, device, input_size)
        print('    Complete model loaded!')
        if args.mode == 'complete':
            owners = branch_targets(model, 'complete')
            nets = model
        else:
            saved = SAVED_NAME_MAP.get(args.backbone, args.backbone)
            experts = {
                'dm': load_base_model(args.backbone, os.path.join(
                    args.models_dir, 'dm_generated', f'{saved}.pt'), device),
                'real': load_base_model(args.backbone, os.path.join(
                    args.models_dir, 'real', f'{saved}.pt'), device),
            }
            owners = branch_targets(experts, 'per-block')
            nets = experts

        print('\n[2] Running prediction...')
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        prob_dm, prob_real = float(probs[0, 0]), float(probs[0, 1])
        pred_class = int(probs.argmax())
        print(f'    Prediction: {CLASS_NAMES[pred_class]}')
        print(f'    Probabilities: DM={prob_dm:.4f}, REAL={prob_real:.4f}')

        # Which expert actually drove this decision? Exact up to bias terms.
        attribution = branch_attribution(model, x, pred_class)
        print(f"    Branch attribution: DM={attribution['attr_dm']:+.3f}  "
              f"REAL={attribution['attr_real']:+.3f}  "
              f"(logit={attribution['logit']:+.3f}, "
              f"residual={attribution['residual']:+.3f})")

        if args.mode == 'per-block':
            # In per-block mode class 1 means "belongs to this block's own
            # class" (see umbalanced_dataset). The index is right; what was
            # wrong before was showing a map for a hypothesis the expert scores
            # at p~0.001 without saying so.
            targets = [1]
            names = {1: 'own-class'}
            for key, net in nets.items():
                with torch.no_grad():
                    p = float(torch.softmax(net(x), dim=1)[0, 1])
                print(f'    {key} expert p(own class) = {p:.4f}'
                      + ('   <-- LOW EVIDENCE, map not interpretable' if p < 0.05 else ''))
        else:
            targets = [0, 1] if args.all_targets else [pred_class]
            names = CLASS_NAMES

    labels = [lbl for lbl, _, _ in owners]

    # --- attribution --------------------------------------------------------
    print('\n[3] Computing attribution maps...')
    results = run_attribution(owners, x, targets, stages, methods, args)

    metrics = {
        'image': args.image_path,
        'input_size': input_size,
        'mode': 'sanity' if args.sanity else args.mode,
        'prediction': CLASS_NAMES.get(pred_class, str(pred_class)),
        'prob_dm': prob_dm,
        'prob_real': prob_real,
        'branch_attribution': attribution,
        'maps': {f'{s}|{m}|{l}|{k}': dict(results[(s, m, l, k)].meta)
                 for (s, m, l, k) in results},
    }

    # --- V4 flip-equivariance gate -----------------------------------------
    print('\n[4] Flip-equivariance check (V4)...')
    print('    Compares CAM(flip(x)) against flip(CAM(x)).')
    print('    Read it COMPARATIVELY, not against an absolute threshold: a')
    print('    correct implementation on ImageNet-resnet50 measures ~+0.66 on a')
    print('    turtle photo but ~-0.02 on a near-symmetric bowl of apples, where')
    print('    the flipped image legitimately makes a different object salient.')
    print('    What is diagnostic is the POPULATION mean over many images versus')
    print('    the same score for the shared prior map - see diagnose_prior.py T2.')
    metrics['flip_equivariance'] = {}
    for stage in stages:
        print(f'    stage={stage}')
        metrics['flip_equivariance'][stage] = flip_equivariance(
            owners, x, targets[0], stage, 'gradcam', args)

    # --- input-space attribution -------------------------------------------
    extra = {}
    if args.occlusion:
        print('\n[5] Occlusion map (reference)...')
        net = owners[0][1]
        extra['occlusion'] = occlusion_map(net, x, targets[0],
                                           baseline=args.scorecam_baseline,
                                           batch_size=args.batch_size)
        print(f"    positions={extra['occlusion'].meta['n_positions']}")
    if args.integrated_gradients:
        print('\n[6] Integrated gradients...')
        net = owners[0][1]
        extra['integrated_gradients'] = integrated_gradients(
            net, x, targets[0], baseline=args.scorecam_baseline)
        r = extra['integrated_gradients'].meta['completeness_residual']
        print(f'    completeness residual = {r:.3%}'
              + ('   <-- FAILS the 5% tolerance (V3)' if r > 0.05 else '   (V3 ok)'))
    for name, res in extra.items():
        metrics['maps'][f'input|{name}'] = dict(res.meta)

    # --- figures ------------------------------------------------------------
    print('\n[7] Saving visualizations...')
    if args.sanity:
        title = (f'V1 SANITY GATE - ImageNet resnet50, class {pred_class}\n'
                 'if these maps do not land on the object, the bug is in the '
                 'attribution code')
    else:
        title = (f'Explainability - Prediction: {class_label(pred_class)} '
                 f'(DM={prob_dm:.3f}, REAL={prob_real:.3f})')
        if args.mode == 'complete':
            title += "\nmaps attribute THIS model's logits"
    figure_main(img_np, results, stages, methods, labels, targets[0],
                os.path.join(out_dir, 'explanation_comparison.png'), title)
    figure_signed_and_scale(results, stages, methods, labels, targets[0],
                            os.path.join(out_dir, 'explanation_signed.png'))

    if extra:
        fig, axes = plt.subplots(1, 1 + len(extra),
                                 figsize=(3.9 * (1 + len(extra)), 4.3),
                                 squeeze=False, layout='constrained')
        axes[0][0].imshow(img_np)
        axes[0][0].set_title('Original', fontsize=9)
        axes[0][0].axis('off')
        for ax, (name, res) in zip(axes[0][1:], extra.items()):
            draw_panel(ax, img_np, res, f'{name}\n(input space)')
        path = os.path.join(out_dir, 'explanation_input_space.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'    Saved: {path}')

    # raw maps, so nothing downstream has to re-derive them from a PNG
    np.savez_compressed(
        os.path.join(out_dir, 'maps.npz'),
        **{f'{s}|{m}|{l}|{k}': results[(s, m, l, k)].raw for (s, m, l, k) in results},
        **{f'input|{n}': r.raw for n, r in extra.items()})

    # --- robustness ---------------------------------------------------------
    if not args.no_robustness and not args.sanity:
        metrics['robustness'] = run_robustness(
            model, owners, img, transform, device, results, stages,
            pred_class, targets[0], img_np, out_dir, args)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"    Saved: {os.path.join(out_dir, 'metrics.json')}")

    print('\n' + '=' * 62)
    print('EXPLANATION COMPLETE - 2 BLOCKS (DM + REAL)')
    print('=' * 62)
    print(f'Prediction: {class_label(pred_class)}')
    if not args.sanity:
        print(f'Probabilities: DM={prob_dm:.4f}, REAL={prob_real:.4f}')
    print(f'Results saved in: {out_dir}/')
    print('=' * 62)


def run_robustness(model, owners, img, transform, device, baseline_results,
                   stages, original_pred, target, img_np, out_dir, args):
    """16-transform sweep. ECS is measured on the NATIVE grid, never upsampled."""
    print('\n[8] Robustness analysis...')
    rng_seed = args.seed
    transformations = [
        ('JPEG QF90', jpeg_compression(90)),
        ('JPEG QF75', jpeg_compression(75)),
        ('JPEG QF50', jpeg_compression(50)),
        ('JPEG QF25', jpeg_compression(25)),
        ('Blur r=0.5', gaussian_blur(0.5)),
        ('Blur r=1.0', gaussian_blur(1.0)),
        ('Blur r=2.0', gaussian_blur(2.0)),
        ('Resize 0.75x', resize_down_up(0.75)),
        ('Resize 0.5x', resize_down_up(0.5)),
        ('Resize 0.25x', resize_down_up(0.25)),
        ('Brightness +20%', brightness_adjust(1.2)),
        ('Brightness -20%', brightness_adjust(0.8)),
        ('Contrast +30%', contrast_adjust(1.3)),
        ('Sharpness +100%', sharpness_adjust(2.0)),
        ('Color +20%', color_adjust(1.2)),
        ('Screenshot', screenshot_simulation(seed=rng_seed)),
    ]

    stage = args.robustness_stage
    rows = []
    for name, fn in transformations:
        pil_t = fn(img)
        xt = transform(pil_t).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(torch.softmax(model(xt), dim=1).argmax())

        row = {'name': name, 'stable': int(pred == original_pred),
               'pred': CLASS_NAMES[pred]}
        for label, net, owner in owners:
            layer = resolve_stage(owner, stage)
            res = compute_cam(net, xt, target, layer, method='gradcam', score=args.score)
            ref = baseline_results[(stage, 'gradcam', label, target)]
            r, reason = ecs_pearson(ref.map, res.map)
            iou, iou_reason = ecs_iou(ref.map, res.map)
            key = label.split('-')[0].lower()
            row[f'ecs_{key}'] = r
            row[f'ecs_reason_{key}'] = reason
            row[f'iou_{key}'] = iou
            row[f'iou_reason_{key}'] = iou_reason
            row[f'degenerate_{key}'] = res.degenerate
        rows.append(row)
        print(f"    {name:<18} " + '  '.join(
            f"{k}={row[k]:+.3f}" if isinstance(row.get(k), float) and not np.isnan(row[k])
            else f"{k}=nan" for k in row if k.startswith('ecs_') and not k.startswith('ecs_reason')))

    _figure_robustness(rows, owners, os.path.join(out_dir, 'robustness_analysis.png'))

    summary = {'stage': stage, 'rows': rows,
               'stable': sum(r['stable'] for r in rows), 'n': len(rows)}
    for label, _, _ in owners:
        key = label.split('-')[0].lower()
        vals = [r[f'ecs_{key}'] for r in rows if not np.isnan(r[f'ecs_{key}'])]
        summary[f'mean_ecs_{key}'] = float(np.mean(vals)) if vals else float('nan')
        summary[f'n_degenerate_{key}'] = sum(r[f'degenerate_{key}'] for r in rows)
        print(f"    Mean ECS {key.upper()}: {summary[f'mean_ecs_{key}']:+.3f} "
              f"({summary[f'n_degenerate_{key}']} degenerate maps excluded)")
    print(f"    Stable predictions: {summary['stable']}/{summary['n']}")
    return summary


def _figure_robustness(rows, owners, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    names = [r['name'] for r in rows]
    y = np.arange(len(names))
    width = 0.8 / max(len(owners), 1)
    for i, (label, _, _) in enumerate(owners):
        key = label.split('-')[0].lower()
        vals = [0.0 if np.isnan(r[f'ecs_{key}']) else r[f'ecs_{key}'] for r in rows]
        ax1.barh(y + (i - (len(owners) - 1) / 2) * width, vals, width, label=label)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Explainability Consistency Score (Pearson, native grid)')
    # Was xlim(0, 1), which silently clipped every negative correlation - one of
    # the reasons an anti-correlated map could never be noticed.
    ax1.set_xlim(-1, 1)
    ax1.axvline(0, color='k', lw=1)
    ax1.axvline(0.7, color='green', ls='--', alpha=0.5)
    ax1.axvline(0.3, color='red', ls='--', alpha=0.5)
    ax1.set_title('Robustness - Grad-CAM ECS')
    ax1.legend()

    stable = sum(r['stable'] for r in rows)
    ax2.bar(['Stable', 'Changed'], [stable, len(rows) - stable], color=['green', 'red'])
    ax2.set_ylim(0, len(rows))
    ax2.set_title(f'Prediction Stability ({stable}/{len(rows)} stable)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'    Saved: {path}')


if __name__ == '__main__':
    main()
