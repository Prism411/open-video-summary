"""
Deep feature strategies for quality scoring (DINOv2, ResNet-50).

Shared scoring framework: L2 norm of mean features per frame (Option A).
Optional lambda penalty for temporal inconsistency: score = mean_norm - lambda * std_norm.
"""

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from cv2 import cvtColor, COLOR_BGR2RGB
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy


_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DeepFeatureStrategy(ScoringStrategy):
    """Base for deep feature quality scoring (L2 norm of mean features)."""

    def __init__(
        self,
        batch_size: int = 32,
        device: str | None = None,
        use_fp16: bool = True,
        lambda_penalty: float = 0.0,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.lambda_penalty = lambda_penalty
        self._model: nn.Module | None = None

    @abstractmethod
    def _load_model(self) -> nn.Module:
        ...

    @abstractmethod
    def _extract_batch_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from preprocessed batch -> (B, feat_dim)."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        ...

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
            self._model.eval()
            self._model.to(self.device)
            if self.use_fp16:
                self._model.half()
        return self._model

    @staticmethod
    def _preprocess_frames(frames: list) -> torch.Tensor:
        """BGR frames -> (N, 3, 224, 224) normalized tensor."""
        return torch.stack([
            _IMAGENET_TRANSFORM(cvtColor(f, COLOR_BGR2RGB)) for f in frames
        ])

    def extract_features(self, frames: list) -> ndarray:
        if not frames:
            return np.zeros((0, self.feature_dim))

        tensor = self._preprocess_frames(frames)
        all_features = []

        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                batch = tensor[i : i + self.batch_size].to(self.device)
                if self.use_fp16:
                    batch = batch.half()
                features = self._extract_batch_features(batch)
                all_features.append(features.float().cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def score_segments(
        self, segment_features: dict[VideoSegment, ndarray]
    ) -> list[tuple[VideoSegment, float]]:
        scores = []
        for segment, features in segment_features.items():
            if len(features) == 0:
                scores.append((segment, 0.0))
                continue

            per_frame_norms = np.linalg.norm(features, axis=1)
            mean_norm = float(np.mean(per_frame_norms))
            std_norm = float(np.std(per_frame_norms))
            score = mean_norm - self.lambda_penalty * std_norm
            scores.append((segment, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    @property
    def requires_color(self) -> bool:
        return True

    def release_model(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Released model from GPU memory.")


class DINOv2Strategy(DeepFeatureStrategy):
    """DINOv2 ViT-S/14 (self-supervised) — patch token mean for quality scoring.

    Uses mean of patch tokens (trained via iBOT masked-image-modeling) rather
    than the CLS token (which has KoLeo regularization that homogenizes norms).
    """

    def __init__(self, model_name: str = "dinov2_vits14_reg", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name

    def _load_model(self) -> nn.Module:
        log.info(f"Loading DINOv2 model: {self.model_name}")
        return torch.hub.load("facebookresearch/dinov2", self.model_name)

    def _extract_batch_features(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(batch)
        patch_tokens = output["x_norm_patchtokens"]  # (B, N_patches, 384)
        return patch_tokens.mean(dim=1)  # (B, 384)

    @property
    def feature_dim(self) -> int:
        return 384


class ResNet50Strategy(DeepFeatureStrategy):
    """ResNet-50 (ImageNet supervised) — GAP features for quality scoring.

    Uses Global Average Pooling features (2048-D) from the penultimate layer.
    Weights: IMAGENET1K_V2 (improved training recipe from TorchVision).
    """

    def _load_model(self) -> nn.Module:
        from torchvision.models import resnet50, ResNet50_Weights

        log.info("Loading ResNet-50 (ImageNet supervised)")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()  # Remove classifier -> output is 2048-D GAP features
        return model

    def _extract_batch_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # (B, 2048)

    @property
    def feature_dim(self) -> int:
        return 2048
