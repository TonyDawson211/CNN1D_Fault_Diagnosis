import torch
import random


class TimeSeriesAugment:
    """
    对 1D 振动信号的轻量增强：
    - 随机时移（shift_ratio）
    - 轻噪声（noise_std）
    - 幅度缩放（scale_range）
    - 时间遮挡（mask_ratio）
    只在训练集调用
    """

    def __init__(self, shift_ratio=0.1, noise_std=0.01, scale_range: tuple = (0.9, 1.1), mask_ratio=0.05):
        self.shift_ratio = shift_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor):
        length = x.size(-1)
        # 随机时移
        if random.random() < 0.5:
            max_shift = int(length * self.shift_ratio)
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=-1)
        # 轻噪声
        if random.random() < 0.5:
            std = x.std() + 1e-8
            x += self.noise_std * std * torch.randn_like(x)
        # 幅度缩放
        if random.random() < 0.5:
            zoom_scale = random.uniform(*self.scale_range)
            x *= zoom_scale
        # 时间遮挡
        if random.random() < 0.5:
            mask_len = max(1, int(self.mask_ratio * length))
            start = random.randint(0, length - mask_len)
            x[:, start:start + mask_len] = 0

        return x


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
