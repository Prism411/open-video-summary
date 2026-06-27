from numpy import ndarray

from open_video_summary.entities.video import VideoSegment
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor
from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy


class SIFTBoVWStrategy(ScoringStrategy):
    """Original SIFT + Bag-of-Visual-Words scoring (Barbieri 2021)."""

    def __init__(self, dict_size: int = 300) -> None:
        self.dict_size = dict_size

    def extract_features(self, frames: list) -> ndarray:
        return ImageProcessor.ks_sift(frames)

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
        return False
