"""
IQA-based scoring strategies: LIQE and NIQE.

Uses pyiqa library for No-Reference Image Quality Assessment.
These replace SIFT+BoVW in Stage 4 with actual quality metrics.

LIQE (Zhang, CVPR 2023): Multi-task CLIP — predicts quality, distortion, scene.
  Higher score = better quality. Requires GPU.

NIQE (Mittal et al., 2013): Natural Scene Statistics features, no training labels.
  Lower score = better quality. Works on CPU.
"""

import gc

import cv2
import numpy as np
import torch
from numpy import ndarray

from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy


class _PyIQAStrategy(ScoringStrategy):
    """Base for pyiqa-backed IQA strategies."""

    metric_name: str = ""
    lower_is_better: bool = False

    def __init__(self, device: str | None = None, batch_size: int = 1) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import pyiqa
            log.info(f"Loading {self.metric_name} on {self._device}")
            self._model = pyiqa.create_metric(self.metric_name, device=self._device)
        return self._model

    @property
    def requires_color(self) -> bool:
        return True

    def _frame_to_tensor(self, frame) -> torch.Tensor:
        """Single BGR uint8 frame -> (3, H, W) float32 [0, 1]."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def extract_features(self, frames: list) -> ndarray:
        """Per-frame IQA scores as 1-D feature vector.

        Processes frames in batches to avoid OOM on long segments.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)

        all_scores = []
        for i in range(0, len(frames), self._batch_size):
            chunk = frames[i : i + self._batch_size]
            batch = torch.stack([self._frame_to_tensor(f) for f in chunk])
            batch = batch.to(self._device)
            with torch.inference_mode():
                scores = self.model(batch)
            all_scores.append(scores.flatten().cpu().numpy())
            del batch, scores

        return np.concatenate(all_scores).astype(np.float32)

    def score_segments(
        self, segment_features: dict[VideoSegment, ndarray]
    ) -> list[tuple[VideoSegment, float]]:
        results = []
        for segment, frame_scores in segment_features.items():
            if len(frame_scores) == 0:
                results.append((segment, 0.0))
                continue
            mean_score = float(np.mean(frame_scores))
            results.append((segment, mean_score))

        # Sort: higher=better for LIQE, lower=better for NIQE/BRISQUE
        return sorted(results, key=lambda x: x[1], reverse=not self.lower_is_better)

    def release_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info(f"Released {self.metric_name} from memory.")


class LIQEStrategy(_PyIQAStrategy):
    """LIQE — Learned Image Quality Evaluator (Zhang, CVPR 2023).

    Multi-task CLIP: predicts quality level, distortion type, and scene.
    Higher score = better quality. Batch-capable.
    """

    metric_name = "liqe"
    lower_is_better = False

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device, batch_size=8)


class NIQEStrategy(_PyIQAStrategy):
    """NIQE — Natural Image Quality Evaluator (Mittal et al., 2013).

    Completely blind (no training labels). Lower score = better quality.
    Works on CPU. Processes one frame at a time.
    """

    metric_name = "niqe"
    lower_is_better = True

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device or "cpu", batch_size=1)
