import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROPHESEE_DEFAULT_WIDTH = 1280
PROPHESEE_DEFAULT_HEIGHT = 720
PROPHESEE_DEFAULT_WINDOW_US = 50_000


@dataclass(frozen=True)
class PropheseeWindow:
    recording_id: str
    window_start: int
    window_end: int
    events: np.ndarray
    boxes: np.ndarray


@dataclass(frozen=True)
class PropheseeWindowMetadata:
    recording_id: str
    split: str
    dat_path: str
    label_path: str
    window_start: int
    window_end: int
    bbox_ts: int
    event_left: int
    event_right: int
    num_events: int
    boxes: list
    width: int
    height: int

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, payload):
        return cls(
            recording_id=str(payload["recording_id"]),
            split=str(payload["split"]),
            dat_path=str(payload["dat_path"]),
            label_path=str(payload["label_path"]),
            window_start=int(payload["window_start"]),
            window_end=int(payload["window_end"]),
            bbox_ts=int(payload["bbox_ts"]),
            event_left=int(payload["event_left"]),
            event_right=int(payload["event_right"]),
            num_events=int(payload["num_events"]),
            boxes=[list(row) for row in payload.get("boxes", [])],
            width=int(payload.get("width", PROPHESEE_DEFAULT_WIDTH)),
            height=int(payload.get("height", PROPHESEE_DEFAULT_HEIGHT)),
        )


def read_label_map(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload[str(idx)] for idx in sorted(int(key) for key in payload))


def read_dataset_label_map(root):
    root = Path(root)
    path = root / "label_map_dictionary.json"
    if path.exists():
        return read_label_map(path)
    return tuple(str(idx) for idx in range(7))


def _skip_prophesee_dat_header(handle):
    event_type = 0
    event_size = 8
    metadata = {}

    while True:
        pos = handle.tell()
        line = handle.readline()
        if not line:
            raise ValueError("DAT file ended before event payload")
        if not line.startswith(b"% "):
            handle.seek(pos)
            event_type_bytes = handle.read(1)
            event_size_bytes = handle.read(1)
            if event_type_bytes and event_size_bytes:
                event_type = int(np.frombuffer(event_type_bytes, dtype=np.uint8)[0])
                event_size = int(np.frombuffer(event_size_bytes, dtype=np.uint8)[0])
            if event_type not in (0, 12):
                raise ValueError(f"Unsupported Prophesee event type {event_type}; expected Event2D/EventCD")
            return event_size, metadata

        text = line[2:].decode("utf-8", errors="ignore").strip()
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"width", "height"}:
            try:
                metadata[parts[0].lower()] = int(parts[1])
            except ValueError:
                pass


def read_prophesee_dat(path, width=None, height=None):
    """Read a Prophesee .dat file into Nx4 [x, y, t, p] float32 events."""
    with open(path, "rb") as handle:
        event_size, metadata = _skip_prophesee_dat_header(handle)
        if event_size != 8:
            raise ValueError(f"Unsupported event size {event_size} in {path}; expected 8 bytes")
        raw = np.fromfile(handle, dtype=[("t", "<u4"), ("data", "<u4")])

    width = int(width or metadata.get("width", PROPHESEE_DEFAULT_WIDTH))
    height = int(height or metadata.get("height", PROPHESEE_DEFAULT_HEIGHT))
    timestamps = raw["t"].astype(np.uint64)
    packed = raw["data"]
    x = (packed & 0x3FFF).astype(np.float32)
    y = ((packed >> 14) & 0x3FFF).astype(np.float32)
    polarity = ((packed >> 28) & 0x1).astype(np.float32)
    polarity = polarity * 2.0 - 1.0

    events = np.stack([x, y, timestamps.astype(np.float32), polarity], axis=1)
    valid = (events[:, 0] >= 0) & (events[:, 0] < width)
    valid &= (events[:, 1] >= 0) & (events[:, 1] < height)
    return np.ascontiguousarray(events[valid].astype(np.float32))


def infer_sensor_size_from_dat(path):
    with open(path, "rb") as handle:
        _, metadata = _skip_prophesee_dat_header(handle)
    return (
        int(metadata.get("width", PROPHESEE_DEFAULT_WIDTH)),
        int(metadata.get("height", PROPHESEE_DEFAULT_HEIGHT)),
    )


def load_prophesee_boxes(path):
    boxes = np.load(path)
    required = {"ts", "x", "y", "w", "h", "class_id"}
    if boxes.dtype.names is None or not required.issubset(boxes.dtype.names):
        raise ValueError(f"Labels must be a structured npy with fields {sorted(required)}")
    return boxes


def clip_boxes_xywh(boxes, width, height, num_classes):
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    class_id = boxes["class_id"].astype(np.float32)
    x1 = np.clip(boxes["x"].astype(np.float32), 0, width - 1)
    y1 = np.clip(boxes["y"].astype(np.float32), 0, height - 1)
    x2 = np.clip(boxes["x"].astype(np.float32) + boxes["w"].astype(np.float32), 0, width)
    y2 = np.clip(boxes["y"].astype(np.float32) + boxes["h"].astype(np.float32), 0, height)
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)
    keep = (w > 0) & (h > 0) & (class_id >= 0) & (class_id < num_classes)
    return np.stack([class_id[keep], x1[keep], y1[keep], w[keep], h[keep]], axis=1).astype(np.float32)


def slice_events_by_index(events, start_ts, left, right):
    window = np.ascontiguousarray(events[left:right].copy(), dtype=np.float32)
    if len(window):
        window[:, 2] -= float(start_ts)
    return window


def discover_prophesee_files(root, split):
    root = Path(root)
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Could not find split {split!r} under {root}")
    pairs = []
    for dat_path in sorted(split_root.rglob("*_td.dat")):
        candidates = [
            dat_path.with_name(dat_path.name.replace("_td.dat", "_bbox.npy")),
            dat_path.with_name(dat_path.stem + "_bbox.npy"),
            dat_path.with_suffix(".npy"),
        ]
        label_path = next((path for path in candidates if path.exists()), None)
        if label_path is not None:
            pairs.append((dat_path.resolve(), label_path.resolve()))
    return pairs


def build_prophesee_window_index(root, split, window_us=PROPHESEE_DEFAULT_WINDOW_US, max_files=None):
    root = Path(root)
    class_names = read_dataset_label_map(root)
    pairs = discover_prophesee_files(root, split)
    if max_files is not None:
        pairs = pairs[: int(max_files)]

    entries = []
    for dat_path, label_path in pairs:
        width, height = infer_sensor_size_from_dat(dat_path)
        events = read_prophesee_dat(dat_path, width=width, height=height)
        boxes = load_prophesee_boxes(label_path)
        timestamps = events[:, 2] if len(events) else np.zeros((0,), dtype=np.float32)
        for ts in np.unique(boxes["ts"]):
            window_end = int(ts)
            window_start = max(0, window_end - int(window_us))
            event_left = int(np.searchsorted(timestamps, window_start, side="left"))
            event_right = int(np.searchsorted(timestamps, window_end, side="left"))
            clipped = clip_boxes_xywh(
                boxes[boxes["ts"] == ts],
                width=width,
                height=height,
                num_classes=len(class_names),
            )
            entries.append(
                PropheseeWindowMetadata(
                    recording_id=dat_path.stem,
                    split=split,
                    dat_path=str(dat_path),
                    label_path=str(label_path),
                    window_start=window_start,
                    window_end=window_end,
                    bbox_ts=window_end,
                    event_left=event_left,
                    event_right=event_right,
                    num_events=max(event_right - event_left, 0),
                    boxes=clipped.astype(np.float32).tolist(),
                    width=width,
                    height=height,
                )
            )
    return entries


def write_prophesee_window_index(entries, output_path, root=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_root = Path(root).resolve() if root is not None else None
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            payload = entry.to_json()
            if base_root is not None:
                for key in ("dat_path", "label_path"):
                    path = Path(payload[key])
                    try:
                        payload[key] = str(path.resolve().relative_to(base_root))
                    except ValueError:
                        payload[key] = str(path)
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return output_path


def load_prophesee_window_index(index_path, root=None):
    index_path = Path(index_path)
    base_root = Path(root).resolve() if root is not None else None
    entries = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if base_root is not None:
                for key in ("dat_path", "label_path"):
                    path = Path(payload[key])
                    if not path.is_absolute():
                        payload[key] = str((base_root / path).resolve())
            entries.append(PropheseeWindowMetadata.from_json(payload))
    return entries


class PropheseeIndexedWindowDataset:
    def __init__(self, index_path, root=None):
        self.index_path = Path(index_path)
        self.root = Path(root) if root is not None else None
        self.entries = load_prophesee_window_index(self.index_path, root=self.root)
        self._cached_recording_path = None
        self._cached_events = None

    def __len__(self):
        return len(self.entries)

    def _resolve_entry_path(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        if self.root is not None:
            return (self.root / path).resolve()
        return path.resolve()

    def _load_recording_events(self, entry):
        dat_path = self._resolve_entry_path(entry.dat_path)
        if self._cached_recording_path != dat_path:
            self._cached_events = read_prophesee_dat(dat_path, width=entry.width, height=entry.height)
            self._cached_recording_path = dat_path
        return self._cached_events

    def __getitem__(self, idx):
        entry = self.entries[idx]
        events = self._load_recording_events(entry)
        boxes = np.asarray(entry.boxes, dtype=np.float32).reshape(-1, 5)
        return PropheseeWindow(
            recording_id=entry.recording_id,
            window_start=entry.window_start,
            window_end=entry.window_end,
            events=slice_events_by_index(events, entry.window_start, entry.event_left, entry.event_right),
            boxes=boxes,
        )
