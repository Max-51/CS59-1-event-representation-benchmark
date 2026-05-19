import numpy as np
import torch
import torch.nn.functional as F

from src.detection.representations import create_detection_representation


DEFAULT_SENSOR_WIDTH = 1280
DEFAULT_SENSOR_HEIGHT = 720

DEFAULT_DETECTION_METHOD_CONFIGS = {
    "est": {"bins": 9},
    "ergo": {"variant": "event_stack", "stack_size": 12},
    "matrix_lstm": {"hidden_size": 12, "allow_fallback": True},
    "evrepsl": {"variant": "evrep"},
    "get": {"variant": "tokens", "group_num": 12, "patch_size": 4},
    "event_pretraining": {"variant": "group_tokens", "group_num": 12, "patch_size": 4},
    "event_frame": {},
    "event_count": {},
    "binary_event_image": {},
    "timestamp_image": {},
    "time_surface": {},
    "voxel_grid": {"bins": 5},
}


def boxes_to_yolo_xywh(boxes_xywh, width, height):
    if len(boxes_xywh) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    labels = boxes_xywh.astype(np.float32).copy()
    labels[:, 1] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5) / width
    labels[:, 2] = (boxes_xywh[:, 2] + boxes_xywh[:, 4] * 0.5) / height
    labels[:, 3] = boxes_xywh[:, 3] / width
    labels[:, 4] = boxes_xywh[:, 4] / height
    labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)
    return labels


def _as_hwc(tensor, height=DEFAULT_SENSOR_HEIGHT, width=DEFAULT_SENSOR_WIDTH):
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        if arr.shape[:2] == (height, width):
            pass
        elif arr.shape[1:] == (height, width):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[0] > arr.shape[1] and arr.shape[0] > arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] <= arr.shape[0] and arr.shape[2] <= arr.shape[1]:
            pass
        else:
            arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 4:
        if arr.shape[0] == height and arr.shape[1] == width:
            arr = arr.reshape(height, width, -1)
        elif arr.shape[1] == height and arr.shape[2] == width:
            arr = arr.reshape((-1, height, width)).transpose(1, 2, 0)
        else:
            arr = arr[0] if arr.shape[0] == 1 else arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported representation shape: {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def normalize_nonzero(arr):
    arr = arr.astype(np.float32, copy=True)
    mask = arr != 0
    if np.any(mask):
        mean = arr[mask].mean()
        std = arr[mask].std()
        arr[mask] = (arr[mask] - mean) / (std + 1e-6)
    return arr


def resize_hwc_torch(arr, target_hw):
    if arr.shape[:2] == tuple(target_hw):
        return arr
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x.float(), size=target_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0).numpy()


def adapt_channels(arr, target_channels):
    channels = arr.shape[2]
    if channels == target_channels:
        return arr.astype(np.float32)
    if channels < target_channels:
        repeats = int(np.ceil(target_channels / channels))
        return np.tile(arr, (1, 1, repeats))[:, :, :target_channels].astype(np.float32)

    groups = np.array_split(np.arange(channels), target_channels)
    projected = [arr[:, :, group].mean(axis=2) for group in groups]
    return np.stack(projected, axis=2).astype(np.float32)


def letterbox_hwc(arr, new_shape=640, color=0.0):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    height, width = arr.shape[:2]
    scale = min(new_shape[0] / height, new_shape[1] / width)
    resized_hw = (int(round(height * scale)), int(round(width * scale)))
    resized = resize_hwc_torch(arr, resized_hw)
    canvas = np.full((new_shape[0], new_shape[1], arr.shape[2]), color, dtype=np.float32)
    top = (new_shape[0] - resized_hw[0]) // 2
    left = (new_shape[1] - resized_hw[1]) // 2
    canvas[top : top + resized_hw[0], left : left + resized_hw[1]] = resized
    return canvas, scale, (left, top)


def letterbox_yolo_labels(labels, original_hw, scale, pad, new_shape=640):
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    height, width = original_hw
    out = labels.astype(np.float32).copy()
    x_center = labels[:, 1] * width * scale + pad[0]
    y_center = labels[:, 2] * height * scale + pad[1]
    box_w = labels[:, 3] * width * scale
    box_h = labels[:, 4] * height * scale
    out[:, 1] = x_center / new_shape[1]
    out[:, 2] = y_center / new_shape[0]
    out[:, 3] = box_w / new_shape[1]
    out[:, 4] = box_h / new_shape[0]
    out[:, 1:] = np.clip(out[:, 1:], 0.0, 1.0)
    return out


class UnifiedRepresentationAdapter:
    """Method adapter that returns image-like HWC float32 tensors."""

    def __init__(
        self,
        method,
        config=None,
        height=DEFAULT_SENSOR_HEIGHT,
        width=DEFAULT_SENSOR_WIDTH,
        detector_channels=12,
    ):
        base_config = {
            "height": height,
            "width": width,
            "device": "cpu",
            "return_numpy": True,
        }
        base_config.update(DEFAULT_DETECTION_METHOD_CONFIGS.get(method, {}))
        base_config.update(config or {})
        self.method = method
        self.height = height
        self.width = width
        self.detector_channels = detector_channels
        self.representation = create_detection_representation(method, base_config)

    def build_hwc(self, events):
        if len(events) == 0:
            return np.zeros((self.height, self.width, self.detector_channels), dtype=np.float32)
        raw = self.representation.build(events)
        hwc = _as_hwc(raw, height=self.height, width=self.width)
        hwc = resize_hwc_torch(hwc, (self.height, self.width))
        hwc = normalize_nonzero(hwc)
        return hwc.astype(np.float32)

    def build_detector_tensor(self, events):
        hwc = self.build_hwc(events)
        return adapt_channels(hwc, self.detector_channels)


class YoloV6SampleBuilder:
    """Build one YOLOv6-ready training sample from a unified event window."""

    def __init__(
        self,
        method,
        representation_config=None,
        img_size=640,
        detector_channels=12,
        sensor_width=DEFAULT_SENSOR_WIDTH,
        sensor_height=DEFAULT_SENSOR_HEIGHT,
    ):
        self.img_size = img_size
        self.sensor_width = int(sensor_width)
        self.sensor_height = int(sensor_height)
        self.adapter = UnifiedRepresentationAdapter(
            method,
            representation_config,
            height=self.sensor_height,
            width=self.sensor_width,
            detector_channels=detector_channels,
        )

    def build(self, window):
        rep = self.adapter.build_detector_tensor(window.events)
        image, scale, pad = letterbox_hwc(rep, self.img_size)
        labels = boxes_to_yolo_xywh(window.boxes, width=self.sensor_width, height=self.sensor_height)
        labels = letterbox_yolo_labels(
            labels,
            (self.sensor_height, self.sensor_width),
            scale,
            pad,
            self.img_size,
        )

        labels_out = torch.zeros((len(labels), 6), dtype=torch.float32)
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {
            "image": chw,
            "labels": labels_out,
            "recording_id": window.recording_id,
            "window": (window.window_start, window.window_end),
            "original_shape": (self.sensor_height, self.sensor_width),
            "resized_shape": tuple(image.shape[:2]),
        }


def collate_yolov6_samples(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = []
    for batch_idx, item in enumerate(batch):
        sample_labels = item["labels"].clone()
        if len(sample_labels):
            sample_labels[:, 0] = batch_idx
        labels.append(sample_labels)
    labels = torch.cat(labels, dim=0) if labels else torch.zeros((0, 6), dtype=torch.float32)
    paths = [f'{item["recording_id"]}:{item["window"][0]}-{item["window"][1]}' for item in batch]
    shapes = [item["original_shape"] for item in batch]
    return images, labels, paths, shapes
