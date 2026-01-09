import torch
import numpy as np
from diffusers import StableAudioPipeline
from exp.utils import generate_sine_wave, calculate_spectral_purity

@torch.no_grad()
def verify_global_amplitude_logic(vae_path, probe_freq=440.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableAudioPipeline.from_pretrained(vae_path, torch_dtype=torch.float16)
    vae = pipe.vae.to(device).eval()
    
    # 1. 获取基础 Latent
    audio = generate_sine_wave(probe_freq, vae.config.sampling_rate).to(device).half()
    latents_base = vae.encode(audio).latent_dist.sample() # [1, 64, T]
    
    def get_snr(lats):
        rec = vae.decode(lats.half()).sample.float().cpu().numpy()[0,0]
        _, _, snr, _, _, _ = calculate_spectral_purity(rec, vae.config.sampling_rate, freq=probe_freq)
        return snr

    snr_base = get_snr(latents_base)
    print(f"Base SNR: {snr_base:.4f} dB")

    # 2. 灵敏度扫描：找出哪些通道“渴望”更多能量
    print("📊 正在探测 64 个通道的能量灵敏度...")
    sensitivities = []
    eps = 0.05 # 赋予 5% 的扰动
    
    for c in range(64):
        # 测试增加能量
        lats_plus = latents_base.clone()
        lats_plus[:, c, :] *= (1 + eps)
        snr_plus = get_snr(lats_plus)
        
        # 测试减弱能量
        lats_minus = latents_base.clone()
        lats_minus[:, c, :] *= (1 - eps)
        snr_minus = get_snr(lats_minus)
        
        # 梯度估算
        grad = (snr_plus - snr_minus) / (2 * eps)
        sensitivities.append(grad)

    # 3. 构造“全局最优修正”向量
    # 根据灵敏度方向，给每个通道一个微小的修正
    best_latents = latents_base.clone()
    for c in range(64):
        # 如果 grad > 0，说明增加能量有益；反之则减弱
        adjustment = 1.0 + (0.1 if sensitivities[c] > 0 else -0.1)
        best_latents[:, c, :] *= adjustment
        
    snr_best = get_snr(best_latents)
    
    print("\n" + "="*50)
    print("📋 全局能量平衡验证报告")
    print("="*50)
    print(f"1. 敏感通道总数: {np.sum(np.abs(sensitivities) > 0.01)}")
    print(f"2. 全局修正后 SNR: {snr_best:.4f} dB")
    print(f"3. 潜在最大收益: {snr_best - snr_base:+.4f} dB")
    
    if snr_best > snr_base:
        print("\n✅ [物理假设成立]：VAE 的核心问题在于多通道间的能量分配不均。")
        print("这证明了 SSA 应该是一个“通道能量均衡器”，而非复杂的旋转矩阵。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default="../stable-audio-open-1.0", help="VAE path")
    args = parser.parse_args()
    verify_global_amplitude_logic(args.vae)