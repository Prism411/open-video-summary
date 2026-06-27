from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy
from open_video_summary.core.selection_criteria.scoring.sift_bovw import SIFTBoVWStrategy

try:
    from open_video_summary.core.selection_criteria.scoring.deep_features import (
        DINOv2Strategy,
        ResNet50Strategy,
    )
except ImportError:
    pass

try:
    from open_video_summary.core.selection_criteria.scoring.cnn_bovw import (
        CNNBoVWStrategy,
        ResNet18BoVWStrategy,
        ResNet50BoVWStrategy,
        ResNet101BoVWStrategy,
        VGG16BoVWStrategy,
        VGG19BoVWStrategy,
        EfficientNetB0BoVWStrategy,
        MobileNetV3BoVWStrategy,
    )
except ImportError:
    pass

try:
    from open_video_summary.core.selection_criteria.scoring.iqa_strategies import (
        LIQEStrategy,
        NIQEStrategy,
    )
except ImportError:
    pass

__all__ = [
    "ScoringStrategy",
    "SIFTBoVWStrategy",
    "DINOv2Strategy",
    "ResNet50Strategy",
    "CNNBoVWStrategy",
    "ResNet18BoVWStrategy",
    "ResNet50BoVWStrategy",
    "ResNet101BoVWStrategy",
    "VGG16BoVWStrategy",
    "VGG19BoVWStrategy",
    "EfficientNetB0BoVWStrategy",
    "MobileNetV3BoVWStrategy",
    "LIQEStrategy",
    "NIQEStrategy",
]
