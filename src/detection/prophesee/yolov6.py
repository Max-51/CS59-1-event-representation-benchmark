from src.detection.yolov6_common import (
    DEFAULT_DETECTION_METHOD_CONFIGS,
    YoloV6SampleBuilder,
    UnifiedRepresentationAdapter,
    collate_yolov6_samples,
)


DEFAULT_PROPHESEE_METHOD_CONFIGS = DEFAULT_DETECTION_METHOD_CONFIGS


class PropheseeYoloV6SampleBuilder(YoloV6SampleBuilder):
    """YOLOv6 sample builder for Prophesee DAT detection datasets."""

    def __init__(
        self,
        method,
        representation_config=None,
        img_size=320,
        detector_channels=12,
        sensor_width=1280,
        sensor_height=720,
    ):
        super().__init__(
            method=method,
            representation_config=representation_config,
            img_size=img_size,
            detector_channels=detector_channels,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
