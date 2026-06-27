from abc import ABC, abstractmethod
from numpy import ndarray

from open_video_summary.entities.video import VideoSegment


class ScoringStrategy(ABC):
    """Abstract base for quality scoring strategies in Stage 4."""

    @abstractmethod
    def extract_features(self, frames: list) -> ndarray:
        """Extract features from video frames.

        Returns array of shape (N, feat_dim) where N depends on strategy.
        """
        ...

    @abstractmethod
    def score_segments(
        self, segment_features: dict[VideoSegment, ndarray]
    ) -> list[tuple[VideoSegment, float]]:
        """Score and rank segments by visual quality.

        Returns list of (segment, score) sorted descending by score.
        """
        ...

    @property
    def requires_color(self) -> bool:
        """Whether this strategy requires color (BGR) frames."""
        return False
