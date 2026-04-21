import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# N-Caltech101 101个类别，按字母排序
CLASSES = sorted([
    'BACKGROUND_Google', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion',
    'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular',
    'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
    'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier',
    'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile',
    'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
    'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer',
    'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone',
    'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
    'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp',
    'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah',
    'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda',
    'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino',
    'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse',
    'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
    'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch',
    'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'
])

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

class NCaltech101(Dataset):
    def __init__(self, root, split_file, transform=None, max_events=50000):
        """
        root:       数据集根目录（N-Caltech101/）
        split_file: train.txt 或 test.txt 的路径
        transform:  事件表示转换（各方法不同）
        max_events: 截取最多多少个事件（统一长度）
        """
        self.root = Path(root)
        self.transform = transform
        self.max_events = max_events

        # 读取文件路径列表
        self.samples = Path(split_file).read_text().strip().splitlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        full_path = self.root / rel_path

        # 提取类别标签（路径格式：training/accordion/accordion_0.npy）
        class_name = full_path.parent.name
        label = CLASS_TO_IDX[class_name]

        # 加载事件：(N, 4) → [x, y, t, p]
        events = np.load(full_path).astype(np.float32)

        # 截取固定数量事件（太长的样本统一截断）
        if len(events) > self.max_events:
            events = events[:self.max_events]

        # 极性归一化：-1/+1 → 0/1
        events[:, 3] = (events[:, 3] + 1) / 2

        # 转换为指定表示（如voxel grid、time surface等）
        if self.transform:
            events = self.transform(events)

        return events, label
