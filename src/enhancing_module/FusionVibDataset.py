import torch

from src.enhancing_config.TimeSeriesAugment import TimeSeriesAugment


class FusionVibDataset:
    """
    TenserDataset的强化数据集VibDataset
    """

    def __init__(self, x_np_1d, x_np_2d, y_np, is_augmented: bool):
        self.x_1d = x_np_1d
        self.x_2d = x_np_2d
        self.y = y_np
        self.is_augmented = is_augmented
        self.augmenter = TimeSeriesAugment() if is_augmented else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_1d = torch.as_tensor(self.x_1d[item]).float().clone()
        x_2d = torch.as_tensor(self.x_2d[item]).float().clone()
        y = torch.tensor(int(self.y[item]))
        # 数据增强
        if self.augmenter:
            x_1d = self.augmenter(x_1d)

        return x_1d, x_2d, y
