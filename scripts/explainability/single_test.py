import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import types

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dfx.architecture import completenn
from dfx.import_classifiers import backbone
from dfx.explainability import (
    CompleteModelExplainer,
    RobustnessAnalyzer,
    visualize_explanation,
    overlay_heatmap,
    jpeg_compression,
    gaussian_blur,
    resize_down_up,
    screenshot_simulation,
    brightness_adjust,
    contrast_adjust,
    sharpness_adjust,
    color_adjust,
    tensor_to_pil,
    pil_to_tensor
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepFeatureX Explainability Tool - Bypass Combiner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python %(prog)s --models_dir ../working_dir/models --approach_dir unbalancing-approach \
      --backbone efficientnet_b0 --image_path path/to/image.png

  # Empiric positive_idx test (recommended for first run)
  python %(prog)s --models_dir ../working_dir/models --approach_dir unbalancing-approach \
      --backbone efficientnet_b0 --image_path img.png \
      --test_dm_known ../datasets/dm_generated/dm_001.png \
      --test_real_known ../datasets/real/real_001.png
        """
    )

    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--approach_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['efficientnet_b0', 'efficientnet_b4',
                                 'efficientnet_widese_b0', 'efficientnet_widese_b4',
                                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnext101',
                                 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'])
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../explanation_results')
    parser.add_argument('--positive_idx', type=int, default=1,
                        help='Ositive class index in Base Models (0 or 1). '
                             'Use --test_dm_known e --test_real_known to verify it.')
    parser.add_argument('--test_dm_known', type=str, default=None,
                        help='Known DM image for empiric positive_idx test')
    parser.add_argument('--test_real_known', type=str, default=None,
                        help='Known REAL image for empiric positive_idx test')

    return parser.parse_args()


def load_base_model(backbone_name: str, model_path: str, device: torch.device):
    model = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    if 'efficientnet' in backbone_name:
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    elif 'resnet' in backbone_name or 'resnext' in backbone_name:
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
    elif 'densenet' in backbone_name:
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    elif 'vit' in backbone_name:
        model.heads = nn.Identity()

    model = model.to(device)
    model.eval()
    return model


def load_base_model_classifier(backbone_name: str, model_path: str, device: torch.device):
    model = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def load_complete_model_custom(
    backbone_name: str,
    models_dir: str,
    approach_dir: str,
    device: torch.device,
    positive_idx: int = 1
):
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
    saved_name = saved_name_map.get(backbone_name, backbone_name)

    dm_path = os.path.join(models_dir, approach_dir, 'dm_generated', f'{saved_name}.pt')
    gan_path = os.path.join(models_dir, approach_dir, 'gan_generated', f'{saved_name}.pt')
    real_path = os.path.join(models_dir, approach_dir, 'real', f'{saved_name}.pt')

    for path, name in [(dm_path, 'DM'), (gan_path, 'GAN'), (real_path, 'REAL')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} model not found: {path}")
        print(f"    Found {name} model: {path}")

    # Load versions WITH classifier (for prediction)
    print("    Loading DM Base Model (classifier)...")
    model_dm_cls = load_base_model_classifier(backbone_name, dm_path, device)
    print("    Loading GAN Base Model (classifier)...")
    model_gan_cls = load_base_model_classifier(backbone_name, gan_path, device)
    print("    Loading REAL Base Model (classifier)...")
    model_real_cls = load_base_model_classifier(backbone_name, real_path, device)

    # Load versions WITHOUT classifier (for heatmap)
    print("    Loading DM Base Model (feature extractor)...")
    model_dm_feat = load_base_model(backbone_name, dm_path, device)
    print("    Loading GAN Base Model (feature extractor)...")
    model_gan_feat = load_base_model(backbone_name, gan_path, device)
    print("    Loading REAL Base Model (feature extractor)...")
    model_real_feat = load_base_model(backbone_name, real_path, device)

    # ========================================================================
    # COMBINER BYPASS
    # ========================================================================

    # 1. Creates completenn with feature extractors (as in original code)
    complete = completenn(model_dm_feat, model_gan_feat, model_real_feat)
    complete = complete.to(device)

    # 2. Adds binary classifiers as attributes of completenn
    complete.dm_cls = model_dm_cls
    complete.gan_cls = model_gan_cls
    complete.real_cls = model_real_cls
    complete.positive_idx = positive_idx

    # 3. Overrides forward with the bypass
    def bypass_forward(self, x):
        logit_dm = self.dm_cls(x)[:, self.positive_idx]
        logit_gan = self.gan_cls(x)[:, self.positive_idx]
        logit_real = self.real_cls(x)[:, self.positive_idx]
        return torch.stack([logit_dm, logit_gan, logit_real], dim=1)

    complete.forward = types.MethodType(bypass_forward, complete)
    complete.eval()

    return complete


def test_positive_idx(model_dm_cls, model_gan_cls, model_real_cls,
                      img_dm_path, img_real_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_probs(model, path):
        img = Image.open(path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = torch.softmax(model(t), dim=1)
        return out[0].cpu().numpy()

    print("\n" + "="*60)
    print("Empiric positive_idx test...")
    print("="*60)

    for name, model in [('DM', model_dm_cls), ('GAN', model_gan_cls), ('REAL', model_real_cls)]:
        p_dm_img = get_probs(model, img_dm_path)
        p_real_img = get_probs(model, img_real_path)

        print(f"\n{name} Base Model:")
        print(f"  On DM image:     idx0={p_dm_img[0]:.4f}, idx1={p_dm_img[1]:.4f}")
        print(f"  On REAL image:   idx0={p_real_img[0]:.4f}, idx1={p_real_img[1]:.4f}")

        if name == 'DM':
            guess = 0 if p_dm_img[0] > p_dm_img[1] else 1
            print(f"  -> DM model: classe {guess} = DM (probabilmente)")
        elif name == 'REAL':
            guess = 0 if p_real_img[0] > p_real_img[1] else 1
            print(f"  -> REAL model: classe {guess} = REAL (probabilmente)")

    print("="*60)


def main():
    # ========================================================================
    # 0. PARSE COMMAND LINE ARGUMENTS
    # ========================================================================
    args = parse_args()

    MODELS_DIR = args.models_dir
    APPROACH_DIR = args.approach_dir
    BACKBONE = args.backbone
    IMAGE_PATH = args.image_path
    OUTPUT_DIR = args.output_dir
    POSITIVE_IDX = args.positive_idx

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # 1. LOAD MODEL WITH BYPASS
    # ========================================================================
    print("\n[1] Loading DeepFeatureX model (BYPASS mode)...")
    print(f"    Models dir: {MODELS_DIR}")
    print(f"    Approach: {APPROACH_DIR}")
    print(f"    Backbone: {BACKBONE}")
    print(f"    positive_idx: {POSITIVE_IDX}")

    model = load_complete_model_custom(BACKBONE, MODELS_DIR, APPROACH_DIR, device, POSITIVE_IDX)
    print(f"    Bypass model loaded successfully!")

    # Test empirico positive_idx se richiesto
    if args.test_dm_known and args.test_real_known:
        # Carica temporaneamente i modelli con classificatore per il test
        saved_name_map = {
            'efficientnet_b0': 'effb0', 'efficientnet_b4': 'effb4',
            'resnet50': 'res50', 'densenet121': 'dense121',
        }
        saved_name = saved_name_map.get(BACKBONE, BACKBONE)
        dm_path = os.path.join(MODELS_DIR, APPROACH_DIR, 'dm_generated', f'{saved_name}.pt')
        gan_path = os.path.join(MODELS_DIR, APPROACH_DIR, 'gan_generated', f'{saved_name}.pt')
        real_path = os.path.join(MODELS_DIR, APPROACH_DIR, 'real', f'{saved_name}.pt')

        m_dm = load_base_model_classifier(BACKBONE, dm_path, device)
        m_gan = load_base_model_classifier(BACKBONE, gan_path, device)
        m_real = load_base_model_classifier(BACKBONE, real_path, device)
        test_positive_idx(m_dm, m_gan, m_real, args.test_dm_known, args.test_real_known, device)

    # ========================================================================
    # 2. LOAD AND PREPROCESS IMAGE
    # ========================================================================
    print("\n[2] Loading and preprocessing image...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    print(f"    Image shape: {input_tensor.shape}")

    img_np = np.array(img.resize((224, 224)))

    # ========================================================================
    # 3. CREATE EXPLAINER
    # ========================================================================
    print("\n[3] Creating explainability module...")
    explainer = CompleteModelExplainer(model, BACKBONE)
    print("    Explainer ready")

    # ========================================================================
    # 4. GENERATE GRAD-CAM EXPLANATIONS
    # ========================================================================
    print("\n[4] Generating Grad-CAM explanations...")

    gradcam_exp = explainer.full_explanation(input_tensor, method='gradcam')

    print(f"    Prediction: {gradcam_exp['prediction']['class_name']} "
          f"({gradcam_exp['prediction']['confidence']:.2%})")
    print(f"    Probabilities: DM={gradcam_exp['prediction']['probabilities']['dm']:.3f}, "
          f"GAN={gradcam_exp['prediction']['probabilities']['gan']:.3f}, "
          f"REAL={gradcam_exp['prediction']['probabilities']['real']:.3f}")
    print(f"    Attribution: DM={gradcam_exp['attribution']['dm']:.3f}, "
          f"GAN={gradcam_exp['attribution']['gan']:.3f}, "
          f"REAL={gradcam_exp['attribution']['real']:.3f}")

    fig = visualize_explanation(
        img_np,
        gradcam_exp,
        save_path=f'{OUTPUT_DIR}/gradcam_explanation.png'
    )
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR}/gradcam_explanation.png")

    # ========================================================================
    # 5. GENERATE SCORE-CAM EXPLANATIONS
    # ========================================================================
    print("\n[5] Generating Score-CAM explanations...")
    print("    (This may take a while - requires multiple forward passes)")

    scorecam_exp = explainer.full_explanation(input_tensor, method='scorecam')

    fig = visualize_explanation(
        img_np,
        scorecam_exp,
        save_path=f'{OUTPUT_DIR}/scorecam_explanation.png'
    )
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR}/scorecam_explanation.png")

    # ========================================================================
    # 6. COMPARE GRAD-CAM vs SCORE-CAM
    # ========================================================================
    print("\n[6] Comparing Grad-CAM vs Score-CAM...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    for i, key in enumerate(['dm', 'gan', 'real']):
        overlay = overlay_heatmap(img_np, gradcam_exp['heatmaps'][key], alpha=0.5)
        axes[0, i+1].imshow(overlay)
        axes[0, i+1].set_title(f'Grad-CAM: {key.upper()}')
        axes[0, i+1].axis('off')

    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    for i, key in enumerate(['dm', 'gan', 'real']):
        overlay = overlay_heatmap(img_np, scorecam_exp['heatmaps'][key], alpha=0.5)
        axes[1, i+1].imshow(overlay)
        axes[1, i+1].set_title(f'Score-CAM: {key.upper()}')
        axes[1, i+1].axis('off')

    plt.suptitle('Grad-CAM vs Score-CAM Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/comparison_gradcam_scorecam.png', dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/comparison_gradcam_scorecam.png")

    # ========================================================================
    # 7. ROBUSTNESS ANALYSIS
    # ========================================================================
    print("\n[7] Running robustness analysis...")
    print("    Testing explanations under post-processing transformations")

    analyzer = RobustnessAnalyzer(explainer)

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

    print("\n    Running with Grad-CAM...")
    gradcam_results = analyzer.analyze_pipeline(
        input_tensor,
        transformations,
        method='gradcam'
    )

    print("    Running with Score-CAM...")
    scorecam_results = analyzer.analyze_pipeline(
        input_tensor,
        transformations,
        method='scorecam'
    )

    # ========================================================================
    # 8. VISUALIZE ROBUSTNESS RESULTS
    # ========================================================================
    print("\n[8] Visualizing robustness results...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    names = [r['transform_name'] for r in gradcam_results]
    gradcam_ecs = [r['ecs_mean'] for r in gradcam_results]
    scorecam_ecs = [r['ecs_mean'] for r in scorecam_results]

    x = np.arange(len(names))
    width = 0.35

    ax1.barh(x - width/2, gradcam_ecs, width, label='Grad-CAM', color='coral')
    ax1.barh(x + width/2, scorecam_ecs, width, label='Score-CAM', color='skyblue')
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Explainability Consistency Score (ECS)')
    ax1.set_title('Robustness Comparison')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.3, color='red', linestyle='--', alpha=0.5)

    gradcam_stable = sum(1 for r in gradcam_results if not r['prediction_changed'])
    scorecam_stable = sum(1 for r in scorecam_results if not r['prediction_changed'])

    ax2.bar(['Grad-CAM', 'Score-CAM'],
            [gradcam_stable, scorecam_stable],
            color=['coral', 'skyblue'])
    ax2.set_ylabel('Stable Predictions')
    ax2.set_title(f'Prediction Stability (out of {len(transformations)} transforms)')
    ax2.set_ylim(0, len(transformations))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/robustness_analysis.png', dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/robustness_analysis.png")

    print("\n    Robustness Summary:")
    print("    " + "="*70)
    print(f"    {'Transformation':<20} {'Grad-CAM ECS':>12} {'Score-CAM ECS':>12} {'Pred Changed':>12}")
    print("    " + "-"*70)
    for gr, sr in zip(gradcam_results, scorecam_results):
        pred_changed = "YES" if gr['prediction_changed'] else "NO"
        print(f"    {gr['transform_name']:<20} {gr['ecs_mean']:>12.3f} {sr['ecs_mean']:>12.3f} {pred_changed:>12}")
    print("    " + "="*70)

    # ========================================================================
    # 9. PER-BASE-MODEL ECS
    # ========================================================================
    print("\n[9] Saving per-Base-Model ECS breakdown...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, key in enumerate(['dm', 'gan', 'real']):
        gradcam_ecs_key = [r['ecs'][key] for r in gradcam_results]
        scorecam_ecs_key = [r['ecs'][key] for r in scorecam_results]

        x = np.arange(len(names))
        width = 0.35

        axes[idx].barh(x - width/2, gradcam_ecs_key, width, label='Grad-CAM', color='coral')
        axes[idx].barh(x + width/2, scorecam_ecs_key, width, label='Score-CAM', color='skyblue')
        axes[idx].set_yticks(x)
        axes[idx].set_yticklabels(names, fontsize=8)
        axes[idx].set_xlabel('ECS')
        axes[idx].set_title(f'{key.upper()} Base Model')
        axes[idx].set_xlim(0, 1)
        axes[idx].axvline(x=0.7, color='green', linestyle='--', alpha=0.3)
        axes[idx].axvline(x=0.3, color='red', linestyle='--', alpha=0.3)
        if idx == 0:
            axes[idx].legend()

    plt.suptitle('Per-Base-Model Explainability Consistency', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/per_base_model_ecs.png', dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/per_base_model_ecs.png")

    # ========================================================================
    # 10. FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EXPLANATION COMPLETE (BYPASS MODE)")
    print("="*70)
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  - gradcam_explanation.png")
    print(f"  - scorecam_explanation.png")
    print(f"  - comparison_gradcam_scorecam.png")
    print(f"  - robustness_analysis.png")
    print(f"  - per_base_model_ecs.png")
    print("\nKey metrics:")
    print(f"  - Mean Grad-CAM ECS: {np.mean(gradcam_ecs):.3f}")
    print(f"  - Mean Score-CAM ECS: {np.mean(scorecam_ecs):.3f}")
    print(f"  - Grad-CAM stable predictions: {gradcam_stable}/{len(transformations)}")
    print(f"  - Score-CAM stable predictions: {scorecam_stable}/{len(transformations)}")
    print("="*70)
    print(f"Probs: DM={gradcam_exp['prediction']['probabilities']['dm']:.4f}, "
          f"GAN={gradcam_exp['prediction']['probabilities']['gan']:.4f}, "
          f"REAL={gradcam_exp['prediction']['probabilities']['real']:.4f}")


if __name__ == '__main__':
    main()
