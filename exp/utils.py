import scipy.signal as signal

def calculate_lsd(orig, recon, sr):
    """计算对数谱距离 (Log-Spectral Distance)，衡量频谱包络保真度"""
    # 使用与 VAE 训练一致的 STFT 参数
    _, _, S1 = signal.stft(orig, fs=sr, nperseg=1024)
    _, _, S2 = signal.stft(recon, fs=sr, nperseg=1024)
    # 避免对数负无穷
    S1_log = 20 * np.log10(np.abs(S1) + 1e-7)
    S2_log = 20 * np.log10(np.abs(S2) + 1e-7)
    lsd = np.mean(np.sqrt(np.mean((S1_log - S2_log)**2, axis=0)))
    return lsd

def calculate_spectral_flatness(audio):
    """计算谱平坦度 (Spectral Flatness)，数值越低说明频谱越‘干净/尖锐’，越高说明伪影毛刺越多"""
    f, psd = signal.welch(audio, nperseg=1024)
    gmean = np.exp(np.mean(np.log(psd + 1e-7)))
    amean = np.mean(psd)
    return gmean / amean

def generate_sine_wave(freq, sr, duration=1.0):
    """生成用于探测的纯正弦波"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    # 转换为双声道以匹配 VAE 输入 [1, 2, T]
    audio_stereo = torch.cat([audio_t, audio_t], dim=0).unsqueeze(0)
    return audio_stereo

def calculate_metrics(audio_tensor):
    """
    计算 HNR (谐噪比) 和 SNR (信噪比 - 简单估计)
    audio_tensor: [Channels, Length] cpu numpy
    """
    y = audio_tensor[0] # 取左声道
    
    # 1. HNR
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-9)
    
    # 2. RMS Energy (作为简单的信号强度指标)
    rms = np.sqrt(np.mean(y**2))
    
    return {"hnr": hnr, "rms": rms}

def calculate_spectral_purity(audio_np, sr, freq=440.0):
    """分析频谱：主频能量占比、底噪、SNR、THD"""
    n = len(audio_np)
    fft_val = np.fft.rfft(audio_np)
    psd = np.abs(fft_val)**2
    freqs = np.fft.rfftfreq(n, 1/sr)
    
    main_mask = (freqs > freq - 10) & (freqs < freq + 10)
    main_energy = np.sum(psd[main_mask])
    
    harmonic_energy = 0
    noise_mask = np.ones_like(psd, dtype=bool)
    noise_mask &= ~main_mask 
    
    for h in range(2, 11):
        h_freq = freq * h
        if h_freq > sr/2: break
        h_mask = (freqs > h_freq - 20) & (freqs < h_freq + 20)
        harmonic_energy += np.sum(psd[h_mask])
        noise_mask &= ~h_mask
    
    noise_energy = np.sum(psd[noise_mask])
    total_energy = np.sum(psd)
    snr_db = 10 * np.log10(main_energy / (noise_energy + 1e-12))
    thd = harmonic_energy / (main_energy + 1e-12)
    purity = main_energy / (total_energy + 1e-9)
    
    return purity, noise_energy, snr_db, thd, freqs, psd

def plot_heatmap_to_image(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
        
    dim = matrix.shape[0]
    
    # 1. 不再抹除对角线，直接分析全矩阵
    # 计算对角线均值，判断是否接近单位阵 I
    diag_values = np.diagonal(matrix)
    diag_mean = diag_values.mean()
    
    # 2. 优化色阶：为了同时看到“对角线”和“微弱耦合”
    # 我们将 vmax 设为对角线的平均强度，这样非对角线的亮点会以较淡的颜色显示
    vmax = np.percentile(np.abs(matrix), 99.9) 
    # 确保 vmax 至少覆盖 0.1 左右的强度，否则图表会因为自动缩放显得非常嘈杂
    vmax = max(vmax, 0.1)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 RdBu_r 颜色轴：1(深红), 0(白), -1(深蓝)
    im = ax.imshow(matrix, cmap='RdBu_r', interpolation='nearest', 
                   vmin=-vmax, vmax=vmax)
    
    plt.colorbar(im, ax=ax, label='Transformation Strength')
    
    # 标题增加物理诊断信息
    plt.title(f"M Matrix: Orthogonal Rotation Map\nDiag Mean: {diag_mean:.4f} (Target: 1.0) | Max Int: {vmax:.3f}")
    plt.xlabel("Target Channel")
    plt.ylabel("Source Channel")
    
    # 辅助网格线
    ax.set_xticks(np.arange(0, dim, 16)) # 64维模型建议 16 一格
    ax.set_yticks(np.arange(0, dim, 16))
    ax.grid(which="both", color="grey", linestyle=':', linewidth=0.5, alpha=0.3)

    # 渲染导出
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    
    image = Image.open(buf).convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1) 
    return image_tensor