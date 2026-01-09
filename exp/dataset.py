import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class PhilharmoniaVAEDataset(Dataset):
    def __init__(self, data_root, hop_length, sr=44100, latent_seq_len=256, specific_instruments=None):
        """
        specific_instruments: list, 如果不为None，只加载这些乐器的数据（用于验证集构造）
        """
        self.csv_path = os.path.join(data_root, 'meta.csv')
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Meta csv not found at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        # 乐器筛选逻辑
        if specific_instruments is not None and len(specific_instruments) > 0:
            # 过滤只保留特定乐器
            df = df[df['inst'].isin(specific_instruments)]
        
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.sr = sr
        self.hop_length = hop_length
        self.latent_seq_len = latent_seq_len
        
        self.total_samples = self.latent_seq_len * self.hop_length
        self.duration = self.total_samples / self.sr 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_root, row['file'])
        
        audio, _ = librosa.load(path, sr=self.sr, mono=True, duration=self.duration)

        if len(audio) < self.total_samples:
            audio = np.pad(audio, (0, self.total_samples - len(audio)))
        else:
            audio = audio[:self.total_samples]
            
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).repeat(2, 1) # [2, L]
        return {"audio": audio_tensor, "inst": row['inst'], "file": row['file']}