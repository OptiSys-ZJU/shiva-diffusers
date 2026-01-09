import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusers import StableAudioPipeline
from accelerate import Accelerator
from tqdm import tqdm
import time

from exp.models import SSA_Latent_Refiner
from exp.dataset import PhilharmoniaVAEDataset
from exp.utils import calculate_metrics, plot_heatmap_to_image, generate_sine_wave, calculate_spectral_purity, calculate_lsd, calculate_spectral_flatness

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, resolutions=[(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]):
        super().__init__()
        self.resolutions = resolutions
        # 预先生成 window 并注册为 buffer
        for i, (n_fft, hop, win) in enumerate(resolutions):
            self.register_buffer(f'window_{i}', torch.hann_window(win))
            
    # 【改动1】强制禁用 AMP，确保内部全流程 FP32
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y):
        # x, y: [B, C, T]
        B, C, T = x.shape
        
        # 【改动2】先转 float32，再 reshape，最后【必须】加 contiguous()
        # 很多死锁都是因为 reshape 产生的非连续内存导致的
        x_flat = x.float().reshape(B * C, T).contiguous()
        y_flat = y.float().reshape(B * C, T).contiguous()

        # 【改动3】数值保护：防止模型崩坏产生的 NaN/Inf 弄死 STFT 算子
        # VAE 的输出有时候会偶尔跳出极值，导致 FFT 计算卡住
        epsilon = 1e-7
        y_flat = torch.clamp(y_flat, -10.0, 10.0) 
        
        total_loss = 0.0
        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            window = getattr(self, f'window_{i}')
            
            # STFT
            s_x = torch.stft(x_flat, n_fft, hop, win_length=win, window=window, return_complex=True).abs()
            s_y = torch.stft(y_flat, n_fft, hop, win_length=win, window=window, return_complex=True).abs()
            
            # 谱收敛 Loss (Spectral Convergence)
            # 加上 epsilon 防止分母为 0
            sc_loss = torch.norm(s_x - s_y, p="fro") / (torch.norm(s_x, p="fro") + epsilon)
            
            # 对数幅度 Loss (Log-Magnitude)
            # 使用 log10 稍微稳定一些，效果一样
            mag_loss = F.l1_loss(torch.log(s_x + epsilon), torch.log(s_y + epsilon))
            
            total_loss += sc_loss + mag_loss
            
        return total_loss / len(self.resolutions)

def run_validation(refiner, vae, val_loader, cfg, global_step, accelerator, tb_tracker):
    sr = vae.config.sampling_rate
    refiner.eval()
    
    val_metrics = {"hnr_gain": [], "lsd_base": [], "lsd_ssa": [], "lsd": [], "sf_reduction": []}
    
    # 用于记录音频样本的容器
    audio_samples = []
    
    progress_bar = tqdm(
        val_loader, 
        desc="Validation", 
        disable=not accelerator.is_main_process,
        leave=False
    )
    for i, val_batch in enumerate(progress_bar):
        if i >= cfg['validation']['num_val_batches']: break
        v_audio = val_batch['audio'].to(accelerator.device)
        
        with torch.no_grad():
            # 1. 固定采样，确保 Base 和 SSA 面对的是同一个 Latent
            dist = vae.encode(v_audio.to(dtype=torch.float16)).latent_dist
            v_latents = dist.sample() # Sample 一次并固定
            
            v_rec_base = vae.decode(v_latents).sample
            v_refined = refiner(v_latents.to(dtype=torch.float32))
            v_rec_ssa = vae.decode(v_refined.to(dtype=torch.float16)).sample
        
        # 收集第一个 Batch 的音频用于试听
        if i == 0 and accelerator.is_main_process:
            # 取 Batch 里的第一条数据 [C, T]
            audio_samples.append(("Original", v_audio[0].float().cpu()))
            audio_samples.append(("Base_VAE", v_rec_base[0].float().cpu()))
            audio_samples.append(("SSA_Refined", v_rec_ssa[0].float().cpu()))

        if accelerator.is_main_process:
            base_np = v_rec_base.float().cpu().numpy()
            ssa_np = v_rec_ssa.float().cpu().numpy()
            orig_np = v_audio.float().cpu().numpy()
            
            for b in range(v_audio.shape[0]):
                ref_raw = np.array(orig_np[b, 0]).flatten().astype(np.float32)
                base_raw = np.ascontiguousarray(base_np[b, 0], dtype=np.float32)
                ssa_raw = np.ascontiguousarray(ssa_np[b, 0], dtype=np.float32)

                # 1. HNR Gain
                m_b = calculate_metrics(base_raw)
                m_s = calculate_metrics(ssa_raw)
                val_metrics["hnr_gain"].append(m_s['hnr'] - m_b['hnr'])
                
                # 2. LSD (保真度)
                lsd_base = calculate_lsd(ref_raw, base_raw, sr) # Orig vs Base
                lsd_ssa = calculate_lsd(ref_raw, ssa_raw, sr)   # Orig vs SSA
                lsd = calculate_lsd(base_raw, ssa_raw, sr)      # Base vs SSA (核心指标)
                
                val_metrics["lsd_base"].append(lsd_base)
                val_metrics["lsd_ssa"].append(lsd_ssa)
                val_metrics["lsd"].append(lsd)
                
                # 3. Spectral Flatness
                sf_b = calculate_spectral_flatness(base_raw)
                sf_s = calculate_spectral_flatness(ssa_raw)
                val_metrics["sf_reduction"].append((sf_b - sf_s) / (sf_b + 1e-9))

    if accelerator.is_main_process:
        # 1. 记录音频样本 (Audio Samples)
        for tag, audio_tensor in audio_samples:
            # add_audio 需要 [1, T] 或 [T]
            # audio_tensor 是 [2, T] (Stereo)，我们取 mean 变成单声道方便试听，或者保留 Stereo
            # Tensorboard通常支持 [C, T]，但也可能需要 [1, T]。保险起见转单声道试听。
            # 如果想听立体声，确保 add_audio 参数 sample_rate 正确
            tb_tracker.writer.add_audio(f"val_samples/{tag}", audio_tensor, global_step, sample_rate=sr)

        # 2. 记录 LSD 组合曲线 (Scalars)
        # 这会将三条线画在同一个图里，方便对比 Trend
        tb_tracker.writer.add_scalars("val/LSD_Analysis", {
            "Base_vs_Orig": np.mean(val_metrics["lsd_base"]),
            "SSA_vs_Orig": np.mean(val_metrics["lsd_ssa"]),
            "SSA_vs_Base": np.mean(val_metrics["lsd"])
        }, global_step)

        tb_tracker.writer.add_scalar("val/sf_reduction_pct", np.mean(val_metrics["sf_reduction"]) * 100, global_step)

        # 3. 物理频点探测 (Probe)
        probe_freqs = [220.0, 440.0, 880.0]
        probe_results = []
        fig, axes = plt.subplots(len(probe_freqs), 1, figsize=(10, 4 * len(probe_freqs)))
        if len(probe_freqs) == 1: axes = [axes]

        for idx, freq in enumerate(probe_freqs):
            probe_audio = generate_sine_wave(freq, sr).to(accelerator.device).to(dtype=torch.float16)
            with torch.no_grad():
                # 同样的逻辑，采样一次
                dist = vae.encode(probe_audio).latent_dist
                p_latents = dist.sample()
                
                p_rec_base = vae.decode(p_latents).sample.float().cpu().numpy()[0, 0]
                p_refined_lat = refiner(p_latents.to(dtype=torch.float32))
                p_rec_ssa = vae.decode(p_refined_lat.to(dtype=torch.float16)).sample.float().cpu().numpy()[0, 0]
            
            pur_b, n_b, snr_b, thd_b, f_axis, psd_b = calculate_spectral_purity(p_rec_base, sr, freq=freq)
            pur_s, n_s, snr_s, thd_s, _, psd_s = calculate_spectral_purity(p_rec_ssa, sr, freq=freq)
            
            snr_gain = snr_s - snr_b
            noise_red = (n_b - n_s) / (n_b + 1e-9) * 100
            probe_results.append({'freq': freq, 'snr_gain': snr_gain, 'noise_red': noise_red})

            axes[idx].semilogy(f_axis, psd_b, label=f'Base (SNR:{snr_b:.1f}dB)', alpha=0.5, color='gray')
            axes[idx].semilogy(f_axis, psd_s, label=f'SSA (SNR:{snr_s:.1f}dB)', alpha=0.8, color='red')
            axes[idx].set_xlim(0, 5000)
            axes[idx].set_title(f"Probe {freq}Hz | SNR Gain: {snr_gain:+.3f}dB | Noise Red: {noise_red:.2f}%")
            axes[idx].legend()

        avg_snr_gain = np.mean([r['snr_gain'] for r in probe_results])
        avg_noise_red = np.mean([r['noise_red'] for r in probe_results])
        
        tb_tracker.writer.add_scalar("val_physics/avg_snr_gain_db", avg_snr_gain, global_step)
        tb_tracker.writer.add_scalar("val_physics/avg_noise_reduction_pct", avg_noise_red, global_step)
        
        plt.tight_layout()
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img = plt.imread(buf)
        tb_tracker.writer.add_image("val_physics/spectrum_sweep", img, global_step, dataformats='HWC')

    if accelerator.is_main_process:
        avg_total_gain = np.mean(val_metrics["hnr_gain"]) if val_metrics["hnr_gain"] else 0
        tb_tracker.writer.add_scalar("val/avg_hnr_gain_total", avg_total_gain, global_step)
        
        unwrapped_model = accelerator.unwrap_model(refiner)
        m_img = plot_heatmap_to_image(unwrapped_model.rotation.weight.detach().cpu().numpy())
        tb_tracker.writer.add_image("val/m_heatmap", m_img, global_step)
        
        print(f"✨ Step {global_step} | HNR Gain: {avg_total_gain:.2f}% | SNR Gain: {avg_snr_gain:.3f}dB | LSD(vs Base): {np.mean(val_metrics['lsd']):.4f}")

    refiner.train()

def main():
    parser = argparse.ArgumentParser(description="Train SSA Latent Refiner")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration json file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=cfg['output_dir'],
        mixed_precision=cfg['training'].get('mixed_precision', 'fp16')
    )
    
    if accelerator.is_main_process:
        os.makedirs(cfg['output_dir'], exist_ok=True)
        with open(os.path.join(cfg['output_dir'], 'experiment_config.json'), 'w') as f:
            json.dump(cfg, f, indent=4)
        
        # 简单处理 config 扁平化
        def flatten_config(config, parent_key='', sep='.'):
            items = []
            for k, v in config.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            return dict(items)
            
        flat_cfg = flatten_config(cfg)
        accelerator.init_trackers("ssa_experiment", config=flat_cfg)
        
        print(f"🚀 Training started on {accelerator.device}")

    pipe = StableAudioPipeline.from_pretrained(cfg['pretrained_model'], torch_dtype=torch.float16)
    vae = pipe.vae.to(accelerator.device)
    vae.requires_grad_(False)
    
    sr = vae.config.sampling_rate
    hop_length = np.prod(vae.config.downsampling_ratios)
    vae_channels = vae.config.decoder_input_channels
    
    if accelerator.is_main_process:
        print(f"📊 Model Config: SR={sr}, Hop={hop_length}, Channels={vae_channels}")

    train_dataset = PhilharmoniaVAEDataset(
        cfg['data_root'], 
        hop_length, 
        sr=sr, 
        latent_seq_len=cfg['training']['latent_seq_len']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True # 建议加上 drop_last 防止梯度累积在最后一个batch出错
    )
    
    val_dataset = PhilharmoniaVAEDataset(
        cfg['data_root'], 
        hop_length, 
        sr=sr, 
        latent_seq_len=cfg['training']['latent_seq_len'],
        specific_instruments=cfg['validation']['instruments']
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=2)

    refiner = SSA_Latent_Refiner(channels=vae_channels)
    
    # 使用分层学习率
    optimizer = torch.optim.AdamW([
        {'params': refiner.rotation.parameters(), 'lr': 1e-4},
        {'params': refiner.context_net.parameters(), 'lr': 5e-5}
    ], weight_decay=0.01)

    refiner, optimizer, train_loader, val_loader = accelerator.prepare(
        refiner, optimizer, train_loader, val_loader
    )

    global_step = 0
    start_epoch = 0
    resume_path = cfg['training'].get('resume_from_checkpoint')
    
    if resume_path:
        # Resume 逻辑保持不变...
        if resume_path == "latest":
            dirs = [d for d in os.listdir(cfg['output_dir']) if d.startswith("checkpoint")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                resume_path = os.path.join(cfg['output_dir'], dirs[-1])
            else:
                resume_path = None
        
        if resume_path and os.path.exists(resume_path):
            accelerator.print(f"🔄 Resuming training from {resume_path}")
            accelerator.load_state(resume_path)
            try:
                global_step = int(os.path.basename(resume_path).split("_")[1])
                start_epoch = global_step // len(train_loader)
            except:
                pass

    # 初始化多分辨率 Loss
    mr_stft_loss = MultiResolutionSTFTLoss().to(accelerator.device)

    refiner.train()
    for epoch in range(start_epoch, cfg['training']['num_epochs']): 
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}/{cfg['training']['num_epochs']}")
            
        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(refiner):
                audio = batch['audio']
                
                with torch.no_grad():
                    # 采样一次
                    latents = vae.encode(audio.to(dtype=torch.float16)).latent_dist.sample()
                
                # 训练前向
                refined_latents = refiner(latents.to(dtype=torch.float32)) 
                
                # 为了计算Loss，需要解码回音频
                # 注意：这里我们主要依靠 Spectral Loss 来优化
                rec_audio = vae.decode(refined_latents.to(dtype=torch.float16)).sample
                
                # 计算 Loss
                # 1. MSE Loss (权重给低点，如 0.1)
                loss_mse = F.mse_loss(rec_audio.float(), audio.float())

                # 2. Multi-Resolution Spectral Loss (主要优化目标)
                loss_spectral = mr_stft_loss(audio.float(), rec_audio.float())

                # 建议：Spectral 权重加大，MSE 权重减小
                total_loss = loss_mse * cfg['model']['mse_weight'] + cfg['model']['spectral_weight'] * loss_spectral 
                
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(refiner.parameters(), 1.0)
                    
                optimizer.step()

            # --- 同步点 ---
            if accelerator.sync_gradients:
                global_step += 1
                
                if global_step % 50 == 0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(refiner)
                    with torch.no_grad():
                        W = unwrapped_model.rotation.weight
                        # 计算正交误差 W*W.T - I
                        W_eye_dist = torch.norm(torch.matmul(W, W.T) - torch.eye(W.shape[0], device=W.device))

                    logs = {
                        "train/loss_total": total_loss.item(),
                        "train/loss_mse": loss_mse.item(),
                        "train/loss_spectral": loss_spectral.item(),
                        "train/W_orthogonality_error": W_eye_dist.item(),
                        "train/phi_intensity": unwrapped_model.last_phi_intensity,
                        "train/lr": optimizer.param_groups[0]['lr']
                    }
                    accelerator.log(logs, step=global_step)

                # --- Validation Loop ---
                if global_step % cfg['training']['val_steps'] == 0:
                    accelerator.wait_for_everyone()
                    tb_tracker = accelerator.get_tracker("tensorboard") if accelerator.is_main_process else None
                    
                    run_validation(
                        refiner, vae, val_loader, cfg, 
                        global_step, accelerator, tb_tracker
                    )
                    accelerator.wait_for_everyone()

                # --- Save Checkpoint ---
                if global_step % cfg['training']['save_steps'] == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg['output_dir'], f"checkpoint_{global_step}")
                        accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    
    accelerator.end_training()

if __name__ == "__main__":
    main()