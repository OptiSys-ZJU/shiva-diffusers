import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations

class SSA_Latent_Refiner(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2

        # 1. 指挥官：增加初始化控制
        self.context_net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.Conv1d(channels, self.half_channels, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.half_channels, self.half_channels),
            nn.Tanh()
        )
        
        # 初始化 phi 输出接近 0，让模型从“不做校准”开始起步
        nn.init.zeros_(self.context_net[-2].weight)
        nn.init.zeros_(self.context_net[-2].bias)

        # 2. 执行算子
        # 使用 Linear 包装正交矩阵
        self.rotation = parametrizations.orthogonal(nn.Linear(channels, channels, bias=False))
        
        # 【重要】初始化旋转矩阵为单位阵，防止训练初期 Latent 被强行扭曲导致 Decoder 崩溃
        with torch.no_grad():
            self.rotation.weight.copy_(torch.eye(channels))

    def forward(self, z):
        B, C, T = z.shape

        # --- A. 感知阶段 ---
        phi = self.context_net(z) 
        self.last_phi_intensity = torch.abs(phi).mean().item()
        
        # 辛增益分配：g_p = e^phi, g_q = e^-phi
        g_p = torch.exp(phi)     # [B, 32]
        g_q = torch.exp(-phi)    # [B, 32]
        g = torch.cat([g_p, g_q], dim=1).unsqueeze(-1) # [B, 64, 1]

        # --- B. 执行阶段 ---
        # 1. 正向投影 (Projection)
        # Linear 层的 forward 实际上是 x @ weight.T
        # 所以 z_rotated = z^T @ W.T = (W @ z)^T
        z_trans = z.transpose(1, 2)  # [B, T, 64]
        z_rotated = self.rotation(z_trans) # 执行旋转
        z_rotated = z_rotated.transpose(1, 2) # 回到 [B, 64, T]

        # 2. 辛对称校准
        z_calibrated = z_rotated * g

        # 3. 逆投影还原 (Back-projection) 【这是之前报错的根源】
        # 你之前的代码用的是 W_mat @ z_calibrated，但在 Linear 层逻辑下这是错的。
        # 正确逻辑：我们需要乘 W 的逆（即 W^T），对于 Linear 来说就是直接乘原始 weight。
        W_mat = self.rotation.weight # [64, 64]
        # 使用 matmul 时，必须确保矩阵左乘的顺序对应基底的变换
        z_final = torch.matmul(W_mat.t(), z_calibrated) 

        return z_final