import torch
import torch.nn as nn


class EAM1D(nn.Module):
    """
    Enhanced Attention Module for 1D (EAM)
    结合了 AvgPooling 和 MaxPooling 的空间注意力机制
    """

    def __init__(self, k=7):
        super().__init__()
        # input: [B, 2, L] -> output: [B, 1, L]
        self.conv = nn.Conv1d(2, 1, k, padding=(k - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        # Channel-wise statistic
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, L]
        a = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        att = self.sigmoid(self.conv(a))  # [B, 1, L]
        return x * att


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    无需降维的通道注意力机制
    """

    def __init__(self, ch, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        y = self.pool(x)  # [B, C, 1]
        # View as [B, 1, C] for 1D Conv over channels
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # [B, C, 1]
        return x * self.sigmoid(y)


class MA1DCNN(nn.Module):
    """
    Multi-Attention 1D CNN
    结构: 6层 Conv-BN-ReLU-EAM-ECA-Pool
    """

    def __init__(self, _, in_channel=1, num_classes=10, input_length=1024):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = in_channel
        # 通道数逐层增加: 1 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
        outs = [16, 32, 64, 128, 256, 512]

        for o in outs:
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, o, 3, padding=1),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                EAM1D(k=7),  # 空间/时间维度注意力
                ECA(o, k=3),  # 通道维度注意力
                nn.MaxPool1d(2)  # 下采样 /2
            ))
            in_ch = o

        # 计算全连接层输入维度
        # 经过6次 MaxPool1d(2)，长度变为 input_length / (2^6)
        # 例如 1024 -> 1024/64 = 16
        reduction_factor = 2 ** len(outs)
        final_len = input_length // reduction_factor
        if final_len < 1:
            raise ValueError(f"Input length {input_length} is too small for 6 downsampling layers.")

        self.flatten_dim = outs[-1] * final_len

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, L]
        # 如果输入是 [B, L]，增加通道维度
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    x = torch.randn(2, 1, 1024)
    model = MA1DCNN(None, in_channel=1, num_classes=10, input_length=1024)
    y = model(x)
    print("Output shape:", y.shape)  # [2, 10]