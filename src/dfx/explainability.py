import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
from typing import Dict, List, Tuple, Optional, Callable
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# HOOK MANAGEMENT

class FeatureExtractorHook:
    """
    Hook to capture intermediate feature maps from any layer.
    Used to extract spatial feature maps before GAP/flattening.
    """
    def __init__(self):
        self.features = None
        self.gradient = None
        self.handle_forward = None
        self.handle_backward = None

    def forward_hook(self, module, input, output):
        """Store the output feature map."""
        self.features = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        """Store the gradient of the output."""
        self.gradient = grad_output[0].detach()

    def register(self, module: nn.Module):
        """Register forward and backward hooks on a module."""
        self.handle_forward = module.register_forward_hook(self.forward_hook)
        self.handle_backward = module.register_full_backward_hook(self.backward_hook)

    def remove(self):
        """Remove all registered hooks."""
        if self.handle_forward is not None:
            self.handle_forward.remove()
        if self.handle_backward is not None:
            self.handle_backward.remove()
        self.features = None
        self.gradient = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# TARGET LAYER RESOLVER

def get_target_layer_for_backbone(model: nn.Module, backbone_name: str) -> nn.Module:
    """
    Identify the last convolutional/feature layer before GAP/flattening
    for different backbone architectures.

    Args:
        model: The backbone model (Base Model)
        backbone_name: Name of the backbone architecture

    Returns:
        The target layer module for hooking
    """
    # ResNet family: last bottleneck in layer4
    if 'resnet' in backbone_name or 'resnext' in backbone_name:
        target = model.layer4[-1]
        return target

    # DenseNet family: last denseblock
    elif 'densenet' in backbone_name:
        target = model.features.denseblock4
        return target

    # EfficientNet family: last feature block
    elif 'efficientnet' in backbone_name:
        target = model.features[-1]
        return target

    # Inception family
    elif 'inception' in backbone_name or 'googlenet' in backbone_name:
        target = model.inception5b
        return target

    # ViT family: last transformer encoder layer
    elif 'vit' in backbone_name:
        target = model.encoder.layers[-1]
        return target

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def get_feature_map_shape(backbone_name: str) -> Tuple[int, int, int]:
    """
    Get expected feature map shape (C, H, W) for each backbone.
    """
    shapes = {
        'resnet18': (512, 7, 7),
        'resnet34': (512, 7, 7),
        'resnet50': (2048, 7, 7),
        'resnet101': (2048, 7, 7),
        'resnet152': (2048, 7, 7),
        'resnext101': (2048, 7, 7),
        'densenet121': (1024, 7, 7),
        'densenet161': (2208, 7, 7),
        'densenet169': (1664, 7, 7),
        'densenet201': (1920, 7, 7),
        'efficientnet_b0': (1280, 7, 7),
        'efficientnet_b4': (1792, 7, 7),
        'efficientnet_widese_b0': (1280, 7, 7),
        'efficientnet_widese_b4': (1792, 7, 7),
        'vit_b_16': (768, 14, 14),
        'vit_b_32': (768, 7, 7),
        'vit_l_16': (1024, 14, 14),
        'vit_l_32': (1024, 7, 7),
    }
    return shapes.get(backbone_name, (2048, 7, 7))


# RESHAPE TRANSFORM FOR ViT

def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """
    Reshape ViT output tokens into spatial feature maps.

    Input:  (batch, 197, embed_dim)  [CLS + 196 patches]
    Output: (batch, embed_dim, height, width)
    """
    result = tensor[:, 1:, :]  # (batch, 196, embed_dim)
    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )
    result = result.permute(0, 3, 1, 2)  # (batch, embed_dim, height, width)
    return result


# GRAD-CAM IMPLEMENTATION

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for DeepFeatureX Base Models.
    """

    def __init__(self, model: nn.Module, backbone_name: str, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.backbone_name = backbone_name
        self.device = next(model.parameters()).device

        if target_layer is None:
            self.target_layer = get_target_layer_for_backbone(model, backbone_name)
        else:
            self.target_layer = target_layer

        self.hook = FeatureExtractorHook()
        self.is_vit = 'vit' in backbone_name

        if self.is_vit:
            patch_size = int(backbone_name.split('_')[-1])
            self.vit_height = 224 // patch_size
            self.vit_width = 224 // patch_size

    def generate(self, input_tensor: torch.Tensor, target_class: int = 0) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index for which to compute CAM

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        input_tensor = input_tensor.to(self.device)

        self.hook.register(self.target_layer)

        try:
            self.model.zero_grad()
            output = self.model(input_tensor)

            features = self.hook.features

            if features is None:
                raise RuntimeError("Hook did not capture features. Check target layer.")

            score = output[0, target_class]
            score.backward(retain_graph=True)

            gradients = self.hook.gradient

            if gradients is None:
                raise RuntimeError("Hook did not capture gradients.")

            if self.is_vit:
                features = vit_reshape_transform(features, self.vit_height, self.vit_width)
                gradients = vit_reshape_transform(gradients, self.vit_height, self.vit_width)

            # Grad-CAM computation
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = torch.sum(weights * features, dim=1)  # (1, H, W)
            cam = F.relu(cam)

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            heatmap = cam.squeeze().cpu().numpy()

            return heatmap

        finally:
            self.hook.remove()

    def generate_batch(self, input_tensors: torch.Tensor, target_classes: List[int]) -> List[np.ndarray]:
        """Generate Grad-CAM for a batch of images."""
        heatmaps = []
        for i, target_class in enumerate(target_classes):
            heatmap = self.generate(input_tensors[i:i+1], target_class)
            heatmaps.append(heatmap)
        return heatmaps


# SCORE-CAM IMPLEMENTATION

class ScoreCAM:
    """
    Score-weighted Class Activation Mapping for DeepFeatureX Base Models.
    Gradient-free method using forward evaluations with masked inputs.
    """

    def __init__(self, model: nn.Module, backbone_name: str, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.backbone_name = backbone_name
        self.device = next(model.parameters()).device

        if target_layer is None:
            self.target_layer = get_target_layer_for_backbone(model, backbone_name)
        else:
            self.target_layer = target_layer

        self.is_vit = 'vit' in backbone_name

        if self.is_vit:
            patch_size = int(backbone_name.split('_')[-1])
            self.vit_height = 224 // patch_size
            self.vit_width = 224 // patch_size

    def _get_feature_maps(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract feature maps using forward hook (no gradients needed)."""
        hook = FeatureExtractorHook()
        hook.register(self.target_layer)

        try:
            with torch.no_grad():
                _ = self.model(input_tensor)
            features = hook.features

            if self.is_vit:
                features = vit_reshape_transform(features, self.vit_height, self.vit_width)

            return features
        finally:
            hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int = 0) -> np.ndarray:
        """
        Generate Score-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index for which to compute CAM

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        input_tensor = input_tensor.to(self.device)

        # 1. Extract feature maps
        with torch.no_grad():
            features = self._get_feature_maps(input_tensor)  # (1, C, H, W)

        # 2. Get baseline score
        baseline_input = torch.zeros_like(input_tensor)
        with torch.no_grad():
            baseline_output = self.model(baseline_input)
        baseline_score = baseline_output[0, target_class].item()

        # 3. For each feature map channel, create masked input and evaluate
        batch_size, n_channels, h, w = features.shape
        scores = []

        batch_size_eval = min(32, n_channels)

        for i in range(0, n_channels, batch_size_eval):
            end_i = min(i + batch_size_eval, n_channels)
            channel_batch = features[0, i:end_i, :, :]

            masks = []
            for c in range(channel_batch.shape[0]):
                ch = channel_batch[c]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min)
                else:
                    ch_norm = torch.zeros_like(ch)
                masks.append(ch_norm)

            masks = torch.stack(masks)

            # Upsample masks to input size
            input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
            masks_upsampled = F.interpolate(
                masks.unsqueeze(1),
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            # Create masked inputs
            masked_inputs = input_tensor * masks_upsampled.unsqueeze(1)

            with torch.no_grad():
                outputs = self.model(masked_inputs)

            batch_scores = outputs[:, target_class].cpu().numpy()
            scores.extend(batch_scores.tolist())

        scores = np.array(scores)

        # 4. Compute weights
        weights = scores - baseline_score

        # 5. Weighted combination
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        features_flat = features[0]

        cam = torch.sum(weights_tensor.view(-1, 1, 1) * features_flat, dim=0)

        # 6. ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()

        return heatmap


# BASE MODEL EXPLAINABILITY WRAPPER

class BaseModelExplainer:
    """
    Wrapper to apply Grad-CAM and Score-CAM to a DeepFeatureX Base Model.
    """

    def __init__(self, base_model: nn.Module, backbone_name: str, model_type: str):
        self.base_model = base_model
        self.backbone_name = backbone_name
        self.model_type = model_type

        self.gradcam = GradCAM(base_model, backbone_name)
        self.scorecam = ScoreCAM(base_model, backbone_name)

    def explain(self, input_tensor: torch.Tensor, method: str = 'gradcam', target_class: int = 0) -> np.ndarray:
        """Generate explanation heatmap."""
        if method == 'gradcam':
            return self.gradcam.generate(input_tensor, target_class)
        elif method == 'scorecam':
            return self.scorecam.generate(input_tensor, target_class)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gradcam' or 'scorecam'.")

    def explain_both(self, input_tensor: torch.Tensor, target_class: int = 0) -> Dict[str, np.ndarray]:
        """Generate both Grad-CAM and Score-CAM heatmaps."""
        return {
            'gradcam': self.gradcam.generate(input_tensor, target_class),
            'scorecam': self.scorecam.generate(input_tensor, target_class)
        }


# COMPLETE MODEL EXPLAINABILITY

class CompleteModelExplainer:
    """
    Explainability for the complete DeepFeatureX model.
    """

    def __init__(self, complete_model: nn.Module, backbone_name: str):
        self.complete_model = complete_model
        self.backbone_name = backbone_name
        self.device = next(complete_model.parameters()).device

        self.dm_explainer = BaseModelExplainer(
            complete_model.model1, backbone_name, 'dm'
        )
        self.gan_explainer = BaseModelExplainer(
            complete_model.model2, backbone_name, 'gan'
        )
        self.real_explainer = BaseModelExplainer(
            complete_model.model3, backbone_name, 'real'
        )

        self.explainers = {
            'dm': self.dm_explainer,
            'gan': self.gan_explainer,
            'real': self.real_explainer
        }

    def explain_base_models(self, input_tensor: torch.Tensor, method: str = 'gradcam') -> Dict[str, np.ndarray]:
        """Generate heatmaps for all 3 Base Models."""
        input_tensor = input_tensor.to(self.device)

        heatmaps = {}
        for name, explainer in self.explainers.items():
            heatmaps[name] = explainer.explain(input_tensor, method=method, target_class=0)

        return heatmaps

    def get_final_attribution(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Determine how much each Base Model contributed to the final decision."""
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            code1 = self.complete_model.model1(input_tensor)
            code2 = self.complete_model.model2(input_tensor)
            code3 = self.complete_model.model3(input_tensor)

            x = torch.cat((code1.unsqueeze(1), code2.unsqueeze(1), code3.unsqueeze(1)), 1)

            conv_output = self.complete_model.convs(x)

            final_output = self.complete_model(input_tensor)
            predicted_class = torch.argmax(final_output, dim=1).item()

            first_conv = self.complete_model.convs[0]

            weights = first_conv.weight.data
            channel_importance = weights.abs().mean(dim=(0, 2)).cpu().numpy()

            total = channel_importance.sum()
            if total > 0:
                attribution = channel_importance / total
            else:
                attribution = np.ones(3) / 3

            return {
                'dm': float(attribution[0]),
                'gan': float(attribution[1]),
                'real': float(attribution[2]),
                'predicted_class': predicted_class,
                'class_names': {0: 'DM', 1: 'GAN', 2: 'REAL'}
            }

    def full_explanation(self, input_tensor: torch.Tensor, method: str = 'gradcam') -> Dict:
        """Complete explanation: heatmaps + attribution + prediction."""
        with torch.no_grad():
            output = self.complete_model(input_tensor.to(self.device))
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        heatmaps = self.explain_base_models(input_tensor, method)
        attribution = self.get_final_attribution(input_tensor)

        return {
            'prediction': {
                'class': pred_class,
                'class_name': ['DM', 'GAN', 'REAL'][pred_class],
                'confidence': confidence,
                'probabilities': {
                    'dm': probs[0, 0].item(),
                    'gan': probs[0, 1].item(),
                    'real': probs[0, 2].item()
                }
            },
            'heatmaps': heatmaps,
            'attribution': attribution,
            'method': method
        }


# PILLOW-BASED VISUALIZATION

def apply_colormap_pil(heatmap: np.ndarray, colormap_name: str = 'jet') -> np.ndarray:
    """
    Apply matplotlib colormap to heatmap using only Pillow/matplotlib.

    Args:
        heatmap: Array (H, W) in [0, 1]
        colormap_name: Name of matplotlib colormap

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    cmap = plt.get_cmap(colormap_name)
    colored = cmap(heatmap)  # Returns RGBA
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored_rgb


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap on original image using Pillow blending.

    Args:
        image: Original image (H, W, 3) in [0, 255] or [0, 1]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Opacity of heatmap

    Returns:
        Overlayed image (H, W, 3) in [0, 255]
    """
    # Normalize image to [0, 255]
    if image.max() <= 1.0:
        image = np.uint8(255 * image)
    else:
        image = np.uint8(image)

    # Resize heatmap to image size if needed
    if heatmap.shape[:2] != image.shape[:2]:
        # Use PIL for resizing
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0

    # Apply colormap
    colored_heatmap = apply_colormap_pil(heatmap)

    # Blend using PIL
    img_pil = Image.fromarray(image)
    heatmap_pil = Image.fromarray(colored_heatmap)

    # PIL blend: alpha * heatmap + (1-alpha) * image
    blended = Image.blend(img_pil, heatmap_pil, alpha)

    return np.array(blended)


def visualize_explanation(
    image: np.ndarray,
    explanation: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Create comprehensive visualization of the explanation.

    Layout:
    - Original image
    - DM heatmap
    - GAN heatmap  
    - REAL heatmap
    - Attribution bar chart
    """
    fig = plt.figure(figsize=figsize)

    # Original image
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Heatmaps
    heatmaps = explanation['heatmaps']
    titles = ['DM Expert', 'GAN Expert', 'REAL Expert']
    colors = ['Reds', 'Blues', 'Greens']

    for i, (key, title, cmap) in enumerate(zip(['dm', 'gan', 'real'], titles, colors)):
        ax = plt.subplot(1, 5, i + 2)

        overlay = overlay_heatmap(image, heatmaps[key], alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f'{title}\n({explanation["method"]})')
        ax.axis('off')

    # Attribution chart
    ax5 = plt.subplot(1, 5, 5)
    attr = explanation['attribution']
    bars = ax5.bar(
        ['DM', 'GAN', 'REAL'],
        [attr['dm'], attr['gan'], attr['real']],
        color=['red', 'blue', 'green'],
        alpha=0.7
    )
    ax5.set_ylim(0, 1)
    ax5.set_title('Base Model Attribution')
    ax5.set_ylabel('Importance')

    # Add prediction info
    pred = explanation['prediction']
    fig.suptitle(
        f"Prediction: {pred['class_name']} ({pred['confidence']:.2%})\n"
        f"Method: {explanation['method']}",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ROBUSTNESS ANALYSIS

class RobustnessAnalyzer:
    """
    Analyze how explanations change under post-processing transformations.
    """

    def __init__(self, explainer: CompleteModelExplainer):
        self.explainer = explainer

    def compute_ecs(self, heatmap1: np.ndarray, heatmap2: np.ndarray) -> float:
        """
        Compute Explainability Consistency Score as IoU of thresholded heatmaps.

        Args:
            heatmap1: Original heatmap
            heatmap2: Transformed heatmap

        Returns:
            ECS in [0, 1]
        """
        # Resize to same size if needed using PIL
        if heatmap1.shape != heatmap2.shape:
            h1, w1 = heatmap1.shape
            h2, w2 = heatmap2.shape
            # Resize heatmap2 to match heatmap1
            img_pil = Image.fromarray(np.uint8(255 * heatmap2))
            img_pil = img_pil.resize((w1, h1), Image.BILINEAR)
            heatmap2 = np.array(img_pil) / 255.0

        # Threshold at 50% of max activation
        thresh1 = heatmap1 > 0.5 * heatmap1.max()
        thresh2 = heatmap2 > 0.5 * heatmap2.max()

        # IoU
        intersection = np.logical_and(thresh1, thresh2).sum()
        union = np.logical_or(thresh1, thresh2).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def analyze_transformation(
        self,
        input_tensor: torch.Tensor,
        transform_fn: Callable,
        method: str = 'gradcam'
    ) -> Dict:
        """Compare explanations before and after a transformation."""
        orig_exp = self.explainer.full_explanation(input_tensor, method)

        transformed = transform_fn(input_tensor.clone())
        trans_exp = self.explainer.full_explanation(transformed, method)

        ecs_scores = {}
        for key in ['dm', 'gan', 'real']:
            ecs = self.compute_ecs(
                orig_exp['heatmaps'][key],
                trans_exp['heatmaps'][key]
            )
            ecs_scores[key] = ecs

        pred_changed = (
            orig_exp['prediction']['class'] != trans_exp['prediction']['class']
        )

        return {
            'ecs': ecs_scores,
            'ecs_mean': np.mean(list(ecs_scores.values())),
            'prediction_changed': pred_changed,
            'original': orig_exp,
            'transformed': trans_exp
        }

    def analyze_pipeline(
        self,
        input_tensor: torch.Tensor,
        transformations: List[Tuple[str, Callable]],
        method: str = 'gradcam'
    ) -> List[Dict]:
        """Run full robustness analysis pipeline."""
        results = []
        for name, transform_fn in transformations:
            result = self.analyze_transformation(input_tensor, transform_fn, method)
            result['transform_name'] = name
            results.append(result)
        return results


# PILLOW-BASED TRANSFORMATIONS

def jpeg_compression(quality: int = 75):
    """Create JPEG compression transform using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor to PIL Image
        img = tensor_to_pil(tensor)

        # Save to BytesIO with JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Load back
        compressed = Image.open(buffer)
        compressed = compressed.convert('RGB')

        return pil_to_tensor(compressed, tensor.device)
    return transform


def gaussian_blur(radius: float = 2.0):
    """Create Gaussian blur transform using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return pil_to_tensor(blurred, tensor.device)
    return transform


def resize_down_up(scale: float = 0.5):
    """Create resize down then up transform using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        w, h = img.size

        # Down
        small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        # Up
        restored = small.resize((w, h), Image.BILINEAR)

        return pil_to_tensor(restored, tensor.device)
    return transform


def screenshot_simulation():
    """Simulate screenshot using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        w, h = img.size

        # Step 1: Slight resize
        displayed = img.resize((w, h), Image.BILINEAR)

        # Step 2: Add slight noise
        img_array = np.array(displayed).astype(np.float32)
        noise = np.random.normal(0, 3, img_array.shape)  # small noise in [0,255] range
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy)

        # Step 3: Compress with JPEG
        buffer = io.BytesIO()
        noisy_img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        compressed = Image.open(buffer)
        compressed = compressed.convert('RGB')

        return pil_to_tensor(compressed, tensor.device)
    return transform


def brightness_adjust(factor: float = 1.2):
    """Adjust brightness using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        return pil_to_tensor(adjusted, tensor.device)
    return transform


def contrast_adjust(factor: float = 1.3):
    """Adjust contrast using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        enhancer = ImageEnhance.Contrast(img)
        adjusted = enhancer.enhance(factor)
        return pil_to_tensor(adjusted, tensor.device)
    return transform


def sharpness_adjust(factor: float = 2.0):
    """Adjust sharpness using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        enhancer = ImageEnhance.Sharpness(img)
        adjusted = enhancer.enhance(factor)
        return pil_to_tensor(adjusted, tensor.device)
    return transform


def color_adjust(factor: float = 1.2):
    """Adjust color saturation using Pillow."""
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor_to_pil(tensor)
        enhancer = ImageEnhance.Color(img)
        adjusted = enhancer.enhance(factor)
        return pil_to_tensor(adjusted, tensor.device)
    return transform


# HELPER FUNCTIONS FOR TENSOR/PIL CONVERSION

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a torch tensor to PIL Image.

    Args:
        tensor: Tensor of shape (1, C, H, W) or (C, H, W), values in [0, 1] or normalized

    Returns:
        PIL Image in RGB
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move to CPU and convert to numpy
    img = tensor.cpu().detach().numpy()

    # If normalized (ImageNet), denormalize
    # Check if values are outside [0,1] range
    if img.min() < 0 or img.max() > 1:
        # Assume ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        img = img * std + mean
        img = np.clip(img, 0, 1)

    # Convert to uint8
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.uint8(np.clip(img * 255, 0, 255))

    return Image.fromarray(img)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Convert PIL Image to torch tensor.

    Args:
        img: PIL Image in RGB
        device: Target device

    Returns:
        Tensor of shape (1, 3, H, W) with ImageNet normalization
    """
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_array = (img_array - mean) / std

    tensor = torch.from_numpy(img_array).float().unsqueeze(0).to(device)
    return tensor
