import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import StableAudioPipeline
from exp.utils import generate_sine_wave, calculate_spectral_purity
import os

@torch.no_grad()
def paper_ready_energy_verification(vae_path, output_dir="paper_plots"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableAudioPipeline.from_pretrained(vae_path, torch_dtype=torch.float16)
    vae = pipe.vae.to(device).eval()
    
    test_freqs = [220, 440, 880, 1500]
    sr = vae.config.sampling_rate
    
    sens_matrix = [] # 存储所有频点的灵敏度向量
    gains = []

    print(f"🔬 正在执行物理验证并生成论文图表...")

    for freq in test_freqs:
        audio = generate_sine_wave(freq, sr, duration=1.0).to(device).half()
        latents_base = vae.encode(audio).latent_dist.sample()
        
        def get_snr(lats):
            rec = vae.decode(lats.half()).sample.float().cpu().numpy()[0,0]
            _, _, snr, _, _, _ = calculate_spectral_purity(rec, sr, freq=freq)
            return snr

        base_snr = get_snr(latents_base)
        
        # 1. 探测灵敏度 (Sensitivity Fingerprint)
        sens_vector = []
        for c in range(64):
            l_plus = latents_base.clone()
            l_plus[:, c, :] *= 1.05
            sens_vector.append(get_snr(l_plus) - base_snr)
        
        sens_vector = np.array(sens_vector)
        sens_matrix.append(sens_vector)
        
        # 2. 模拟修复
        latents_perfect = latents_base.clone()
        for c in range(64):
            direction = 1.05 if sens_vector[c] > 0 else 0.95
            latents_perfect[:, c, :] *= direction
        
        perfect_snr = get_snr(latents_perfect)
        gains.append(perfect_snr - base_snr)
        print(f"   -> [{freq}Hz] Gain: {perfect_snr - base_snr:+.4f} dB")

    # ==================================================
    # 论文图表 1: 频变灵敏度指纹 (Fingerprint Analysis)
    # ==================================================
    plt.figure(figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, freq in enumerate(test_freqs):
        plt.plot(sens_matrix[i], label=f"{freq} Hz", color=colors[i], linewidth=1.5, alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title("Latent Channel SNR Sensitivity Across Frequencies", fontsize=14)
    plt.xlabel("Latent Channel Index", fontsize=12)
    plt.ylabel("SNR Δ (dB) per 5% Gain", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_sensitivity_fingerprint.png", dpi=300)

    # ==================================================
    # 论文图表 2: 灵敏度相关性热力图 (Consistency Proof)
    # 用于证明为什么不能用静态矩阵：不同频率的灵敏度相关性低
    # ==================================================
    plt.figure(figsize=(8, 6))
    consistency_matrix = np.corrcoef(sens_matrix)
    sns.heatmap(consistency_matrix, annot=True, xticklabels=test_freqs, yticklabels=test_freqs, cmap="YlGnBu")
    plt.title("Cross-Frequency Sensitivity Correlation", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_consistency_heatmap.png", dpi=300)

    # ==================================================
    # 论文图表 3: 修复潜力柱状图 (Potential Analysis)
    # ==================================================
    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(f) for f in test_freqs], gains, color='teal', alpha=0.7)
    plt.axhline(np.mean(gains), color='red', linestyle='--', label=f'Avg Gain: {np.mean(gains):.2f}dB')
    plt.title("Theoretical SNR Gain by Energy Re-balancing", fontsize=14)
    plt.ylabel("SNR Gain (dB)", fontsize=12)
    plt.xlabel("Test Frequency (Hz)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_gain_potential.png", dpi=300)

    print(f"\n📈 论文素材已保存至目录: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default="../stable-audio-open-1.0", help="VAE path")
    args = parser.parse_args()
    paper_ready_energy_verification(args.vae)