from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor


class VideoHandler:
    @staticmethod
    def get_segments_until_second(
        video: Video, final_second: int, threshold: int = 1
    ) -> list[VideoSegment]:
        segments = []

        for segment in video.segments:
            if round(segment.end) > (final_second + threshold):
                break
            segments.append(segment)

        return segments
