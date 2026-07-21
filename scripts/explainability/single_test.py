import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dfx.architecture import backbone
from dfx.explainability import (
    jpeg_compression,
    gaussian_blur,
    resize_down_up,
    screenshot_simulation,
    brightness_adjust,
    contrast_adjust,
    sharpness_adjust,
    color_adjust,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepFeatureX Explainability Tool — 2 Blocks (DM + REAL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --models_dir ../working_dir/models --approach_dir unbalancing-approach \
      --backbone resnet50 --image_path path/to/image.png
        """
    )

    parser.add_argument('--models_dir', type=str, required=True,
                        help='Root directory where models are stored')
    parser.add_argument('--approach_dir', type=str, required=True,
                        help='Subdirectory with trained base models (e.g., unbalancing-approach)')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['efficientnet_b0', 'efficientnet_b4',
                                 'efficientnet_widese_b0', 'efficientnet_widese_b4',
                                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnext101',
                                 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'])
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the test image')
    parser.add_argument('--output_dir', type=str, default='../explanation_results',
                        help='Output directory for results')

    return parser.parse_args()


# ==================== UTILITIES ====================
saved_name_map = {
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


def get_target_layer(model, backbone_name):
    """Find the target layer for Grad-CAM / Score-CAM."""
    if 'resnet' in backbone_name or 'resnext' in backbone_name:
        return list(model.layer4.children())[-1]
    elif 'densenet' in backbone_name:
        return model.features.norm5
    elif 'efficientnet' in backbone_name:
        return model.features[-1]
    elif 'vit' in backbone_name:
        raise ValueError("ViT does not support Grad-CAM/Score-CAM (no conv layers)")
    else:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found for Grad-CAM")
        return last_conv


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """Overlay a heatmap on an image."""
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8))
        .resize((img_np.shape[1], img_np.shape[0]))
    )
    cmap = plt.get_cmap('jet')
    heatmap_color = cmap(heatmap_resized / 255.0)[:, :, :3]
    overlay = img_np * (1 - alpha) + heatmap_color * 255 * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_ecs(heatmap1, heatmap2):
    """Explainability Consistency Score (Pearson correlation)."""
    h1 = heatmap1.flatten()
    h2 = heatmap2.flatten()
    if np.std(h1) == 0 or np.std(h2) == 0:
        return 0.0
    return float(np.corrcoef(h1, h2)[0, 1])


# ==================== GRAD-CAM ====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed")

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0].clone()

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap


# ==================== SCORE-CAM ====================
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, target_class):
        self.model.eval()
        activations = None

        def hook(m, inp, out):
            nonlocal activations
            activations = out.detach()

        handle = self.target_layer.register_forward_hook(hook)
        with torch.no_grad():
            _ = self.model(input_tensor)
        handle.remove()

        if activations is None:
            raise RuntimeError("Score-CAM hook failed")

        B, C, H, W = activations.shape
        upsampled = torch.nn.functional.interpolate(
            activations, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )

        mins = upsampled.view(C, -1).min(dim=1, keepdim=True)[0].view(C, 1, 1, 1)
        maxs = upsampled.view(C, -1).max(dim=1, keepdim=True)[0].view(C, 1, 1, 1)
        normalized = (upsampled - mins) / (maxs - mins + 1e-8)

        scores = []
        for i in range(C):
            masked_input = input_tensor * normalized[i:i+1]
            with torch.no_grad():
                out = self.model(masked_input)
            scores.append(out[0, target_class].item())

        scores = torch.tensor(scores).to(input_tensor.device)
        weights = scores.view(C, 1, 1)
        heatmap = torch.sum(activations[0] * weights, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap


# ==================== MODELS ====================
def load_base_model(backbone_name, model_path, device):
    """Load a base model (with classification head) for CAM computation."""
    model = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class CompleteModel2Blocks(nn.Module):
    def __init__(self, backbone_name, models_dir, approach_dir, device):
        super().__init__()
        saved_name = saved_name_map.get(backbone_name, backbone_name)

        dm_path = os.path.join(models_dir, approach_dir, 'dm_generated', f'{saved_name}.pt')
        real_path = os.path.join(models_dir, approach_dir, 'real', f'{saved_name}.pt')

        for path, name in [(dm_path, 'DM'), (real_path, 'REAL')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} base model not found: {path}")
            print(f"    Found {name} base model: {path}")

        base_dm = load_base_model(backbone_name, dm_path, device)
        base_real = load_base_model(backbone_name, real_path, device)

        self.dm_extractor = FeatureExtractor(base_dm).to(device)
        self.real_extractor = FeatureExtractor(base_real).to(device)

        for p in self.dm_extractor.parameters():
            p.requires_grad = False
        for p in self.real_extractor.parameters():
            p.requires_grad = False

        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            feat_dm = self.dm_extractor(dummy)
            feat_real = self.real_extractor(dummy)
            feat_dim = feat_dm.shape[1] + feat_real.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        ).to(device)

    def forward(self, x):
        f_dm = self.dm_extractor(x)
        f_real = self.real_extractor(x)
        features = torch.cat([f_dm, f_real], dim=1)
        return self.classifier(features)


def load_complete_model(backbone_name, models_dir, approach_dir, device):
    """Load the 2-block complete model from models_dir/complete/{saved_name}.pt"""
    saved_name = saved_name_map.get(backbone_name, backbone_name)
    complete_path = os.path.join(models_dir, 'complete', f'{saved_name}.pt')

    print("    Loading 2-block Complete Model...")
    model = CompleteModel2Blocks(backbone_name, models_dir, approach_dir, device)

    if os.path.exists(complete_path):
        state_dict = torch.load(complete_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"    Loaded trained combiner: {complete_path}")
    else:
        raise FileNotFoundError(
            f"Complete model not found: {complete_path}\n"
            f"Please train the complete model first or check the path."
        )

    model = model.to(device)
    model.eval()
    return model


# ==================== MAIN ====================
def main():
    args = parse_args()

    MODELS_DIR = args.models_dir
    APPROACH_DIR = args.approach_dir
    BACKBONE = args.backbone
    IMAGE_PATH = args.image_path
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # ========================================================================
    # 1. LOAD MODELS
    # ========================================================================
    print("\n[1] Loading models...")
    print(f"    Models dir: {MODELS_DIR}")
    print(f"    Approach: {APPROACH_DIR}")
    print(f"    Backbone: {BACKBONE}")

    # Il complete model viene caricato automaticamente da models_dir/complete/
    model_complete = load_complete_model(BACKBONE, MODELS_DIR, APPROACH_DIR, device)
    print(f"    Complete model loaded!")

    # Load base models again (with heads) for CAM explainers
    saved_name = saved_name_map.get(BACKBONE, BACKBONE)
    dm_path = os.path.join(MODELS_DIR, APPROACH_DIR, 'dm_generated', f'{saved_name}.pt')
    real_path = os.path.join(MODELS_DIR, APPROACH_DIR, 'real', f'{saved_name}.pt')

    base_dm = load_base_model(BACKBONE, dm_path, device)
    base_real = load_base_model(BACKBONE, real_path, device)

    # Target layers
    target_dm = get_target_layer(base_dm, BACKBONE)
    target_real = get_target_layer(base_real, BACKBONE)

    gradcam_dm = GradCAM(base_dm, target_dm)
    gradcam_real = GradCAM(base_real, target_real)
    scorecam_dm = ScoreCAM(base_dm, target_dm)
    scorecam_real = ScoreCAM(base_real, target_real)

    # ========================================================================
    # 2. LOAD AND PREPROCESS IMAGE
    # ========================================================================
    print("\n[2] Loading and preprocessing image...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img.resize((224, 224)))

    print(f"    Image shape: {input_tensor.shape}")

    # ========================================================================
    # 3. PREDICTION
    # ========================================================================
    print("\n[3] Running prediction...")

    with torch.no_grad():
        logits = model_complete(input_tensor)
        probs = torch.softmax(logits, dim=1)

    prob_dm = probs[0, 0].item()
    prob_real = probs[0, 1].item()
    pred_class = torch.argmax(probs, dim=1).item()
    pred_name = "DM" if pred_class == 0 else "REAL"

    print(f"    Prediction: {pred_name}")
    print(f"    Probabilities: DM={prob_dm:.4f}, REAL={prob_real:.4f}")

    # ========================================================================
    # 4. GRAD-CAM
    # ========================================================================
    print("\n[4] Generating Grad-CAM...")

    heatmap_dm_grad = gradcam_dm.generate(input_tensor, target_class=1)
    heatmap_real_grad = gradcam_real.generate(input_tensor, target_class=1)

    gradcam_dm.remove_hooks()
    gradcam_real.remove_hooks()

    overlay_dm_grad = overlay_heatmap(img_np, heatmap_dm_grad, alpha=0.5)
    overlay_real_grad = overlay_heatmap(img_np, heatmap_real_grad, alpha=0.5)

    # ========================================================================
    # 5. SCORE-CAM
    # ========================================================================
    print("\n[5] Generating Score-CAM... (this may take a while)")

    heatmap_dm_score = scorecam_dm.generate(input_tensor, target_class=1)
    heatmap_real_score = scorecam_real.generate(input_tensor, target_class=1)

    overlay_dm_score = overlay_heatmap(img_np, heatmap_dm_score, alpha=0.5)
    overlay_real_score = overlay_heatmap(img_np, heatmap_real_score, alpha=0.5)

    # ========================================================================
    # 6. VISUALIZE
    # ========================================================================
    print("\n[6] Saving visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Grad-CAM row
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(overlay_dm_grad)
    axes[0, 1].set_title('Grad-CAM: DM')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(overlay_real_grad)
    axes[0, 2].set_title('Grad-CAM: REAL')
    axes[0, 2].axis('off')

    # Score-CAM row
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_dm_score)
    axes[1, 1].set_title('Score-CAM: DM')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(overlay_real_score)
    axes[1, 2].set_title('Score-CAM: REAL')
    axes[1, 2].axis('off')

    plt.suptitle(f'Explainability — Prediction: {pred_name} (DM={prob_dm:.3f}, REAL={prob_real:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/explanation_comparison.png', dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/explanation_comparison.png")

    # ========================================================================
    # 7. ROBUSTNESS ANALYSIS
    # ========================================================================
    print("\n[7] Running robustness analysis...")

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
        ('Screenshot', screenshot_simulation()),
    ]

    results = []
    original_pred = pred_class

    # Re-instantiate CAMs for robustness
    gradcam_dm_r = GradCAM(base_dm, get_target_layer(base_dm, BACKBONE))
    gradcam_real_r = GradCAM(base_real, get_target_layer(base_real, BACKBONE))

    for name, transform_fn in transformations:
        transformed_pil = transform_fn(img)
        transformed_tensor = transform(transformed_pil).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            out = model_complete(transformed_tensor)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1).item()

        # Grad-CAM on transformed image
        h_dm = gradcam_dm_r.generate(transformed_tensor, target_class=1)
        h_real = gradcam_real_r.generate(transformed_tensor, target_class=1)

        # ECS vs original
        ecs_dm = compute_ecs(heatmap_dm_grad, h_dm)
        ecs_real = compute_ecs(heatmap_real_grad, h_real)

        results.append({
            'name': name,
            'ecs_dm': ecs_dm,
            'ecs_real': ecs_real,
            'stable': 1 if pred == original_pred else 0,
            'pred': pred
        })

    gradcam_dm_r.remove_hooks()
    gradcam_real_r.remove_hooks()

    # ========================================================================
    # 8. VISUALIZE ROBUSTNESS
    # ========================================================================
    print("\n[8] Saving robustness results...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    names = [r['name'] for r in results]
    ecs_dm_vals = [r['ecs_dm'] for r in results]
    ecs_real_vals = [r['ecs_real'] for r in results]
    stable_vals = [r['stable'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax1.barh(x - width/2, ecs_dm_vals, width, label='DM', color='coral')
    ax1.barh(x + width/2, ecs_real_vals, width, label='REAL', color='skyblue')
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Explainability Consistency Score (ECS)')
    ax1.set_title('Robustness — Grad-CAM ECS')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.3, color='red', linestyle='--', alpha=0.5)

    stable_count = sum(stable_vals)
    ax2.bar(['Stable', 'Changed'],
            [stable_count, len(stable_vals) - stable_count],
            color=['green', 'red'])
    ax2.set_ylabel('Count')
    ax2.set_title(f'Prediction Stability ({stable_count}/{len(stable_vals)} stable)')
    ax2.set_ylim(0, len(stable_vals))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/robustness_analysis.png', dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/robustness_analysis.png")

    print("\n    Robustness Summary:")
    print("    " + "="*60)
    print(f"    {'Transformation':<20} {'ECS DM':>10} {'ECS REAL':>10} {'Stable':>8}")
    print("    " + "-"*60)
    for r in results:
        s = "YES" if r['stable'] else "NO"
        print(f"    {r['name']:<20} {r['ecs_dm']:>10.3f} {r['ecs_real']:>10.3f} {s:>8}")
    print("    " + "="*60)

    # ========================================================================
    # 9. FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("EXPLANATION COMPLETE — 2 BLOCKS (DM + REAL)")
    print("="*60)
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - explanation_comparison.png")
    print("  - robustness_analysis.png")
    print("\nKey metrics:")
    print(f"  - Prediction: {pred_name}")
    print(f"  - Probabilities: DM={prob_dm:.4f}, REAL={prob_real:.4f}")
    print(f"  - Mean ECS DM:   {np.mean(ecs_dm_vals):.3f}")
    print(f"  - Mean ECS REAL: {np.mean(ecs_real_vals):.3f}")
    print(f"  - Stable predictions: {stable_count}/{len(results)}")
    print("="*60)


if __name__ == '__main__':
    main()