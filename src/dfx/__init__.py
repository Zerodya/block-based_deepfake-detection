from .architecture import (
    completenn,
    call_saved_model,
    get_complete_model
)
from .dataset_classes import (
    mydataset,
    dataset_for_robustness,
    dataset_for_generaization,
    umbalanced_dataset,
    check_len,
    make_train_valid,
    balance_test,
    balance_binary_test,
    make_binary,
    get_trans
)
# Training helpers pull in wandb / sklearn / IPython. Those are training-only
# dependencies, and requiring them just to compute a saliency map meant the
# explainability scripts could not run on a machine set up only for inference.
# Import them lazily so `from dfx.cam import ...` stays cheap and dependency-free.
try:
    from .training_procedure import (
        training,
        testing
    )
except ImportError as _training_import_error:  # pragma: no cover
    _TRAINING_IMPORT_ERROR = _training_import_error

    def _missing_training_dep(*_args, **_kwargs):
        raise ImportError(
            'dfx training helpers are unavailable because an optional '
            f'training-only dependency is missing: {_TRAINING_IMPORT_ERROR}. '
            'Install the full requirements.txt to train; attribution and '
            'inference do not need them.')

    training = testing = _missing_training_dep
from .dir_paths import get_path
from .import_classifiers import backbone
# Attribution methods. The old `explainability` module was removed: it assumed a
# 3-block / 3-class model that this project no longer trains, hooked headless
# nn.Identity() feature extractors as if they had class logits, and derived its
# "attribution" bar chart from static conv weights, making it identical for every
# image. Its post-processing transforms live on in `postprocessing`.
from .cam import (
    CamResult,
    compute_cam,
    score_cam,
    occlusion_map,
    integrated_gradients,
    branch_attribution,
    assert_gap_linear_degeneracy,
    resolve_stage,
    disable_inplace_relu,
    overlay_heatmap,
    ecs_pearson,
    ecs_iou
)
from .postprocessing import (
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
