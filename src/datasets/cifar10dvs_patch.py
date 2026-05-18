"""
Monkey-patch tonic.datasets.CIFAR10DVS to load from local .aedat files,
bypassing tonic's download logic (which returns 403).
Auto-imported by training scripts via: import src.datasets.cifar10dvs_patch
"""
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import tonic

CLASSES = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9,
}

def _read_aedat(filepath):
    with open(filepath, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break
        raw = np.frombuffer(f.read(), dtype='>u4')
    if len(raw) < 2:
        return np.zeros(0, dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','?')])
    raw  = raw[:(len(raw)//2)*2]
    addr = raw[0::2]
    ts   = raw[1::2].astype(np.uint64)
    dvs  = (addr & (1 << 22)) == 0
    addr, ts = addr[dvs], ts[dvs]
    # Decoder B: verified correct for this dataset
    x = ((addr >> 1) & 0x7F).astype(np.uint16)
    y = ((addr >> 8) & 0x7F).astype(np.uint16)
    p = (addr & 0x1).astype(bool)
    result = np.zeros(len(ts), dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','?')])
    result['t'] = ts
    result['x'] = x
    result['y'] = y
    result['p'] = p
    return result


class CIFAR10DVSDataset(Dataset):
    sensor_size = (128, 128, 2)

    def __init__(self, save_to, transform=None, target_transform=None, **kwargs):
        root = Path(save_to) / 'CIFAR10DVS'
        self.data, self.targets = [], []
        for cls_name, label in sorted(CLASSES.items(), key=lambda x: x[1]):
            for f in sorted((root / cls_name).glob('*.aedat')):
                self.data.append(str(f))
                self.targets.append(label)
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events = _read_aedat(self.data[idx])
        label  = self.targets[idx]
        if self.transform:        events = self.transform(events)
        if self.target_transform: label  = self.target_transform(label)
        return events, label


# Apply patch
tonic.datasets.CIFAR10DVS = CIFAR10DVSDataset
