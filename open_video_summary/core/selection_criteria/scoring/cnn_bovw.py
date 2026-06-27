"""CNN spatial features + BoVW scoring strategies.

Replace ONLY the SIFT feature extractor with CNN spatial features,
keeping the BoVW + TF-IDF scoring pipeline identical to SIFTBoVWStrategy.

Pipeline: frames -> CNN spatial features -> keyframe filter -> BoVW + TF-IDF -> scores
vs SIFT: frames -> SIFT descriptors  -> keyframe filter -> BoVW + TF-IDF -> scores
"""

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray, concatenate
from cv2 import cvtColor, COLOR_BGR2RGB
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from open_video_summary.utils import log
from open_video_summary.entities.image import Keyframe
from open_video_summary.entities.video import VideoSegment
from open_video_summary.handlers.image import KeyframeHandler
from open_video_summary.utils.processing.image import BagOfVisualWords
from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy


_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CNNBoVWStrategy(ScoringStrategy):
    """Base for CNN + BoVW: CNN spatial features scored via BoVW + TF-IDF.

    Replaces only the SIFT feature extraction step.  The BoVW + TF-IDF
    scoring pipeline remains identical to SIFTBoVWStrategy.
    """

    def __init__(
        self,
        dict_size: int = 300,
        batch_size: int = 32,
        device: str | None = None,
        use_fp16: bool = True,
        use_keyframe_filter: bool = True,
    ) -> None:
        self.dict_size = dict_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.use_keyframe_filter = use_keyframe_filter
        self._model: nn.Module | None = None

    # -- abstract hooks for subclasses --

    def _load_model(self) -> nn.Module:
        raise NotImplementedError

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass -> (B, C, H, W) spatial feature maps."""
        raise NotImplementedError

    # -- model lifecycle --

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
            self._model.eval()
            self._model.to(self.device)
            if self.use_fp16:
                self._model.half()
        return self._model

    def release_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Released CNN model from GPU memory.")

    # -- feature extraction (replaces SIFT) --

    def extract_features(self, frames: list) -> ndarray:
        if not frames:
            return np.zeros((0, 128))

        tensors = torch.stack([
            _IMAGENET_TRANSFORM(cvtColor(f, COLOR_BGR2RGB)) for f in frames
        ])

        per_frame_descs: list[ndarray] = []
        with torch.no_grad():
            for i in range(0, len(tensors), self.batch_size):
                batch = tensors[i : i + self.batch_size].to(self.device)
                if self.use_fp16:
                    batch = batch.half()
                spatial = self._extract_spatial_features(batch)  # (B, C, H, W)
                B, C, H, W = spatial.shape
                descs = spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)
                descs = torch.nn.functional.normalize(descs, p=2, dim=2)
                for d in descs.float().cpu().numpy():
                    per_frame_descs.append(d)

        # keyframe filtering — same logic as ImageProcessor.ks_sift
        if self.use_keyframe_filter and len(per_frame_descs) > 2:
            segment_keyframes: list[Keyframe] = []
            for desc in per_frame_descs[1:-1]:
                keyframe = Keyframe(descriptor=desc)
                if KeyframeHandler.is_keyframe(keyframe, segment_keyframes):
                    segment_keyframes.append(keyframe)
            if segment_keyframes:
                return concatenate([kf.descriptor for kf in segment_keyframes])

        # fallback: all frames (no filtering or all filtered out)
        return np.concatenate(per_frame_descs, axis=0)

    # -- scoring (identical to SIFTBoVWStrategy) --

    def score_segments(
        self, segment_features: dict[VideoSegment, ndarray]
    ) -> list[tuple[VideoSegment, float]]:
        bovw = BagOfVisualWords(items=segment_features, dict_size=self.dict_size)
        bovw.fit_kmeans()
        df = bovw.generate_bovw_dataframe()

        scores = df.sum(axis=1).sort_values(ascending=False)
        return [(seg, float(score)) for seg, score in scores.items()]

    @property
    def requires_color(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class ResNet18BoVWStrategy(CNNBoVWStrategy):
    """ResNet-18 spatial features (512-D, 7x7 = 49 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import resnet18, ResNet18_Weights

        log.info("Loading ResNet-18 for BoVW feature extraction")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(model.children())[:-2])

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 512, 7, 7)


class ResNet50BoVWStrategy(CNNBoVWStrategy):
    """ResNet-50 spatial features (2048-D, 7x7 = 49 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import resnet50, ResNet50_Weights

        log.info("Loading ResNet-50 for BoVW feature extraction")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return nn.Sequential(*list(model.children())[:-2])

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 2048, 7, 7)


class ResNet101BoVWStrategy(CNNBoVWStrategy):
    """ResNet-101 spatial features (2048-D, 7x7 = 49 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import resnet101, ResNet101_Weights

        log.info("Loading ResNet-101 for BoVW feature extraction")
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        return nn.Sequential(*list(model.children())[:-2])

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 2048, 7, 7)


class VGG16BoVWStrategy(CNNBoVWStrategy):
    """VGG-16 spatial features (512-D, 14x14 = 196 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import vgg16, VGG16_Weights

        log.info("Loading VGG-16 for BoVW feature extraction")
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        return model.features

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 512, 14, 14)


class VGG19BoVWStrategy(CNNBoVWStrategy):
    """VGG-19 spatial features (512-D, 14x14 = 196 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import vgg19, VGG19_Weights

        log.info("Loading VGG-19 for BoVW feature extraction")
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        return model.features

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 512, 14, 14)


class EfficientNetB0BoVWStrategy(CNNBoVWStrategy):
    """EfficientNet-B0 spatial features (1280-D, 7x7 = 49 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        log.info("Loading EfficientNet-B0 for BoVW feature extraction")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        return model.features

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 1280, 7, 7)


class MobileNetV3BoVWStrategy(CNNBoVWStrategy):
    """MobileNetV3-Large spatial features (960-D, 7x7 = 49 descriptors/frame) + BoVW."""

    def _load_model(self) -> nn.Module:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

        log.info("Loading MobileNetV3-Large for BoVW feature extraction")
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        return model.features

    def _extract_spatial_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 960, 7, 7)
