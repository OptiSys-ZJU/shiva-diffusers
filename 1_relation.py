import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from diffusers import StableAudioPipeline
from exp.utils import generate_sine_wave

@torch.no_grad()
def analyze_signed_subspaces(vae_path, probe_freq=440.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableAudioPipeline.from_pretrained(vae_path, torch_dtype=torch.float16)
    vae = pipe.vae.to(device).eval()
    
    # 1. 获取信号
    sr = vae.config.sampling_rate
    audio = generate_sine_wave(probe_freq, sr, duration=1.0).to(device).half()
    latents = vae.encode(audio).latent_dist.sample().float().cpu().numpy()[0]
    num_channels = latents.shape[0]

    # 2. 计算保留正负号的互相关矩阵
    print(f"🔄 Analyzing {num_channels}x{num_channels} signed correlations...")
    signed_corr_matrix = np.zeros((num_channels, num_channels))
    
    for i in range(num_channels):
        for j in range(num_channels):
            corr = signal.correlate(latents[i], latents[j], mode='same')
            norm = (np.linalg.norm(latents[i]) * np.linalg.norm(latents[j]) + 1e-9)
            
            # 关键修改：寻找绝对值最大的点，但提取该点的原始符号和数值
            abs_max_idx = np.argmax(np.abs(corr))
            signed_corr_matrix[i, j] = corr[abs_max_idx] / norm

    # 3. 绘图：使用 RdBu_r 色盘，0点为白色，1为深红，-1为深蓝
    plt.figure(figsize=(10, 8))
    im = plt.imshow(signed_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title(f"Signed Latent Subspace Correlation (Probe: {probe_freq}Hz)")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")
    plt.colorbar(im, label="Cross-Correlation Coefficient")
    
    plt.tight_layout()
    plt.savefig("latent_signed_subspace.png")
    
    # 4. 统计与分析
    pos_pairs = np.sum(signed_corr_matrix > 0.9) - 64 # 减去对角线自身
    neg_pairs = np.sum(signed_corr_matrix < -0.9)
    
    print("\n" + "="*60)
    print("📋 隐空间【符号相关性】分析报告")
    print("="*60)
    print(f"1. 强正相关通道对 (R > 0.9): {pos_pairs // 2}")
    print(f"2. 强负相关通道对 (R < -0.9): {neg_pairs // 2}")
    print(f"3. 纠缠总对数: {(pos_pairs + neg_pairs) // 2}")

    print("\n🧐 物理逻辑复盘:")
    print("   - 【深蓝色块】(R ≈ -1.0): 这些通道是彼此的镜像，设计目的是为了‘抵消’。")
    print("   - 【深红色块】(R ≈ 1.0): 这些通道是彼此的副本，设计目的是为了‘增强’。")
    print("   - 伪影来源：无论增强还是抵消，只要它们的能量权重（Gain）不匹配，")
    print("     最终叠加时就会产生残差噪声。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default="../stable-audio-open-1.0")
    args = parser.parse_args()
    analyze_signed_subspaces(args.vae)