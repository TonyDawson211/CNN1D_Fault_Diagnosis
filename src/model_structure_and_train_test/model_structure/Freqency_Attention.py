import torch


class Frequency_Attention_Module(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.F1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 1), stride=(2, 1))
        self.F2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.constant_(self.F2.bias, 2.0)
        torch.nn.init.kaiming_normal_(self.F2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, conv_data):
        data_max = torch.max(conv_data, dim=-1, keepdim=True)[0]
        data_max_expanded = data_max.expand_as(conv_data)
        combined = torch.stack([conv_data, data_max_expanded], dim=3)
        interleaved = combined.flatten(-3, -2)
        f = self.F1(interleaved)
        f = self.relu(f)
        ff = self.F2(f)
        w = self.sigmoid(ff)
        augmented_data = conv_data * w
        # print("数据已增强！")
        return augmented_data
