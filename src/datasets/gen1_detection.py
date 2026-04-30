import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


GEN1_WIDTH = 304
GEN1_HEIGHT = 240
GEN1_WINDOW_US = 50_000
GEN1_CLASSES = ("car", "pedestrian")

GEN1_BBOX_DTYPE = np.dtype(
    [
        ("ts", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "u1"),
    ]
)


@dataclass(frozen=True)
class Gen1Window:
    recording_id: str
    window_start: int
    window_end: int
    events: np.ndarray
    boxes: np.ndarray


@dataclass(frozen=True)
class Gen1WindowMetadata:
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
        )


def read_gen1_dat(path):
    """Read a GEN1 .dat event file into Nx4 [x, y, t, p] float32 events."""
    with open(path, "rb") as handle:
        event_size = _skip_prophesee_dat_header(handle)
        if event_size != 8:
            raise ValueError(f"Unsupported GEN1 event size {event_size} in {path}; expected 8 bytes")
        raw = np.fromfile(handle, dtype=[("t", "<u4"), ("data", "<u4")])

    timestamps = raw["t"].astype(np.uint64)
    packed = raw["data"]
    x = (packed & 0x3FFF).astype(np.float32)
    y = ((packed >> 14) & 0x3FFF).astype(np.float32)
    polarity = ((packed >> 28) & 0x1).astype(np.float32)
    polarity = polarity * 2.0 - 1.0

    events = np.stack([x, y, timestamps.astype(np.float32), polarity], axis=1)
    valid = (events[:, 0] >= 0) & (events[:, 0] < GEN1_WIDTH)
    valid &= (events[:, 1] >= 0) & (events[:, 1] < GEN1_HEIGHT)
    return np.ascontiguousarray(events[valid].astype(np.float32))


def _skip_prophesee_dat_header(handle):
    event_type = 0
    event_size = 8

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
            return event_size


def load_gen1_boxes(path):
    boxes = np.load(path)
    required = {"ts", "x", "y", "w", "h", "class_id"}
    if boxes.dtype.names is None or not required.issubset(boxes.dtype.names):
        raise ValueError(f"GEN1 labels must be a structured npy with fields {sorted(required)}")
    return boxes


def clip_boxes_xywh(boxes, width=GEN1_WIDTH, height=GEN1_HEIGHT):
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    class_id = boxes["class_id"].astype(np.float32)
    x1 = np.clip(boxes["x"].astype(np.float32), 0, width - 1)
    y1 = np.clip(boxes["y"].astype(np.float32), 0, height - 1)
    x2 = np.clip(boxes["x"].astype(np.float32) + boxes["w"].astype(np.float32), 0, width)
    y2 = np.clip(boxes["y"].astype(np.float32) + boxes["h"].astype(np.float32), 0, height)
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)
    keep = (w > 0) & (h > 0) & (class_id >= 0) & (class_id < len(GEN1_CLASSES))
    return np.stack([class_id[keep], x1[keep], y1[keep], w[keep], h[keep]], axis=1).astype(np.float32)


def boxes_to_yolo_xywh(boxes_xywh, width=GEN1_WIDTH, height=GEN1_HEIGHT):
    if len(boxes_xywh) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    labels = boxes_xywh.astype(np.float32).copy()
    labels[:, 1] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5) / width
    labels[:, 2] = (boxes_xywh[:, 2] + boxes_xywh[:, 4] * 0.5) / height
    labels[:, 3] = boxes_xywh[:, 3] / width
    labels[:, 4] = boxes_xywh[:, 4] / height
    labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)
    return labels


def slice_events_by_time(events, start_ts, end_ts):
    timestamps = events[:, 2]
    left = np.searchsorted(timestamps, start_ts, side="left")
    right = np.searchsorted(timestamps, end_ts, side="left")
    return slice_events_by_index(events, start_ts, left, right)


def slice_events_by_index(events, start_ts, left, right):
    window = np.ascontiguousarray(events[left:right].copy(), dtype=np.float32)
    if len(window):
        window[:, 2] -= float(start_ts)
    return window


def iter_gen1_windows_from_arrays(
    events,
    boxes,
    recording_id="memory",
    window_us=GEN1_WINDOW_US,
    width=GEN1_WIDTH,
    height=GEN1_HEIGHT,
):
    events = np.asarray(events, dtype=np.float32)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must be Nx4 [x, y, t, p]")

    if len(events):
        order = np.argsort(events[:, 2], kind="stable")
        events = np.ascontiguousarray(events[order], dtype=np.float32)

    unique_ts = np.unique(boxes["ts"])
    timestamps = events[:, 2] if len(events) else np.zeros((0,), dtype=np.float32)
    for ts in unique_ts:
        end_ts = int(ts)
        start_ts = max(0, end_ts - int(window_us))
        left = int(np.searchsorted(timestamps, start_ts, side="left"))
        right = int(np.searchsorted(timestamps, end_ts, side="left"))
        box_group = boxes[boxes["ts"] == ts]
        clipped = clip_boxes_xywh(box_group, width=width, height=height)
        yield Gen1Window(
            recording_id=recording_id,
            window_start=start_ts,
            window_end=end_ts,
            events=slice_events_by_index(events, start_ts, left, right),
            boxes=clipped,
        )


class Gen1DetectionDataset:
    """Original unified preprocessing dataset without caching."""

    SPLIT_DIRS = {
        "train": ("train", "training"),
        "val": ("val", "validation"),
        "test": ("test", "testing"),
    }

    def __init__(self, root, split="train", window_us=GEN1_WINDOW_US):
        self.root = Path(root)
        self.split = split
        self.window_us = int(window_us)
        self.files = discover_gen1_files(self.root, split)
        self.index = self._build_index()

    def _build_index(self):
        index = []
        for file_idx, (_, label_path) in enumerate(self.files):
            boxes = load_gen1_boxes(label_path)
            for ts in np.unique(boxes["ts"]):
                index.append((file_idx, int(ts)))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, ts = self.index[idx]
        dat_path, label_path = self.files[file_idx]
        events = read_gen1_dat(dat_path)
        boxes = load_gen1_boxes(label_path)
        group = boxes[boxes["ts"] == ts]
        clipped = clip_boxes_xywh(group)
        start_ts = max(0, int(ts) - self.window_us)
        return Gen1Window(
            recording_id=dat_path.stem,
            window_start=start_ts,
            window_end=int(ts),
            events=slice_events_by_time(events, start_ts, int(ts)),
            boxes=clipped,
        )


class Gen1IndexedWindowDataset:
    """Dataset backed by a one-time preprocessing index.

    Each metadata record stores the exact event slice boundaries for a 50 ms window.
    At runtime we still read raw events, but we no longer repeat timestamp grouping,
    bbox alignment, or per-window searchsorted work for every method.
    """

    def __init__(self, index_path, root=None):
        self.index_path = Path(index_path)
        self.root = Path(root) if root is not None else None
        self.entries = load_gen1_window_index(self.index_path, root=self.root)
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

    def _load_recording_events(self, dat_path):
        dat_path = self._resolve_entry_path(dat_path)
        if self._cached_recording_path != dat_path:
            self._cached_events = read_gen1_dat(dat_path)
            self._cached_recording_path = dat_path
        return self._cached_events

    def __getitem__(self, idx):
        entry = self.entries[idx]
        events = self._load_recording_events(entry.dat_path)
        boxes = np.asarray(entry.boxes, dtype=np.float32).reshape(-1, 5)
        return Gen1Window(
            recording_id=entry.recording_id,
            window_start=entry.window_start,
            window_end=entry.window_end,
            events=slice_events_by_index(events, entry.window_start, entry.event_left, entry.event_right),
            boxes=boxes,
        )


def _split_root(root, split):
    split_dirs = Gen1DetectionDataset.SPLIT_DIRS
    if split not in split_dirs:
        raise KeyError(f"Unsupported split {split!r}")
    for name in split_dirs[split]:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split {split!r} under {root}")


def discover_gen1_files(root, split):
    root = Path(root)
    split_root = _split_root(root, split)
    dat_files = sorted(split_root.rglob("*.dat"))
    pairs = []
    for dat_path in dat_files:
        candidates = [
            dat_path.with_suffix(".npy"),
            dat_path.with_name(dat_path.stem + "_bbox.npy"),
            dat_path.with_name(dat_path.name.replace("_td.dat", "_bbox.npy")),
            dat_path.parent / "labels" / (dat_path.stem + ".npy"),
        ]
        label_path = next((path for path in candidates if path.exists()), None)
        if label_path is not None:
            pairs.append((dat_path.resolve(), label_path.resolve()))
    return pairs


def build_gen1_window_index(root, split, window_us=GEN1_WINDOW_US, max_files=None):
    root = Path(root)
    pairs = discover_gen1_files(root, split)
    if max_files is not None:
        pairs = pairs[: int(max_files)]

    entries = []
    for dat_path, label_path in pairs:
        events = read_gen1_dat(dat_path)
        boxes = load_gen1_boxes(label_path)
        timestamps = events[:, 2] if len(events) else np.zeros((0,), dtype=np.float32)
        for ts in np.unique(boxes["ts"]):
            window_end = int(ts)
            window_start = max(0, window_end - int(window_us))
            event_left = int(np.searchsorted(timestamps, window_start, side="left"))
            event_right = int(np.searchsorted(timestamps, window_end, side="left"))
            clipped = clip_boxes_xywh(boxes[boxes["ts"] == ts])
            entries.append(
                Gen1WindowMetadata(
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
                )
            )
    return entries


def write_gen1_window_index(entries, output_path, root=None):
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


def load_gen1_window_index(index_path, root=None):
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
            entries.append(Gen1WindowMetadata.from_json(payload))
    return entries
