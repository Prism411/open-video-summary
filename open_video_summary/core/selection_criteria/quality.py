from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.core.selection_criteria.scoring.base import ScoringStrategy
from open_video_summary.core.selection_criteria.scoring.sift_bovw import SIFTBoVWStrategy


class QualityPick(SelectionCriteria):
    def __init__(
        self,
        source_criteria: str,
        top_n_segments: int = 1,
        strategy: ScoringStrategy | None = None,
    ) -> None:
        super().__init__(read_from="pick", source_criteria=source_criteria)
        self.top_n_segments = top_n_segments
        self.strategy = strategy or SIFTBoVWStrategy()

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        clusters = [
            cluster
            for cluster in self.get_criteria_input(handler)
            if isinstance(cluster, set)
        ]
        log.info(f"Retrieved {len(clusters)} cluster to execute {self.name} criteria.")
        log.info(f"Using scoring strategy: {self.strategy.__class__.__name__}")

        for cluster in clusters:
            seg_features = self._extract_cluster_features(cluster)
            ranked = self.strategy.score_segments(seg_features)

            log.info(f"Retrieving top-{self.top_n_segments} segments from cluster.")
            top_segments = [seg for seg, _ in ranked[: self.top_n_segments]]

            for segment in cluster:
                self.discard(handler, segment)
            for segment in top_segments:
                self.include(handler, segment)

        # Release GPU memory if using deep features
        if hasattr(self.strategy, "release_model"):
            self.strategy.release_model()

        return handler

    def _extract_cluster_features(
        self, segments: set[VideoSegment]
    ) -> dict[VideoSegment, list]:
        grayscale = not self.strategy.requires_color
        log.info(
            f"Extracting visual features from {len(segments)} segments "
            f"(grayscale={grayscale})."
        )
        return {
            segment: self.strategy.extract_features(
                VideoProcessor.retrieve_video_frames(
                    segment.video_path,
                    grayscale=grayscale,
                )
            )
            for segment in segments
        }
