import torch

from src.enhancing_config.TimeSeriesAugment import TimeSeriesAugment


class VibDataset:
    """
    TenserDataset的强化数据集VibDataset
    """

    def __init__(self, x_np, y_np, is_augmented: bool):
        self.x = x_np
        self.y = y_np
        self.is_augmented = is_augmented
        self.augmenter = TimeSeriesAugment() if is_augmented else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(int(self.y[item]))
        # 数据增强
        if self.augmenter:
            x = self.augmenter(x)

        return x, y
