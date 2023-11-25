from demucs import pretrained, apply
import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm
import random

# python ./dataset_preprocessing/demucs_dataset.py /host/home/data/MusicDatasets/MIR-1K/UndividedWavfile separated_mir_1K 

device = 'cuda'
model = pretrained.get_model(name="htdemucs").to(device)
model.eval()

def infer_vocal_demucs(mix_np):
    # mix_np is of shape (channels, time)
    mix = torch.tensor([mix_np, mix_np]).float().to(device)
    sources = apply.apply_model(model, mix[None], split=True, overlap=0.5, progress=False)[0]
    return sources[model.sources.index('vocals')].detach().cpu().numpy()


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    separated_dir = sys.argv[2]

    # HT Demucs is actually non-deterministic!
    random.seed(114514)
    np.random.seed(114514)
    torch.manual_seed(114514)
    torch.cuda.manual_seed(114514)

    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)
    
    for audio_name in tqdm(os.listdir(audio_dir)):
        audio_path = os.path.join(audio_dir, audio_name)

        y, _ = librosa.load(audio_path, sr=44100, mono=True)
        output = infer_vocal_demucs(y).T
        output = (output[:,0] + output[:,1]) / 2

        output_path = os.path.join(separated_dir, audio_name)
        sf.write(
            output_path,
            output,
            44100,
            "PCM_16",
        )

'''
Installing with Apt
1. Configure the repository:

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    apt-get update

2. Install the NVIDIA Container Toolkit packages:

apt-get install -y nvidia-container-toolkit

Configuring Docker
Configure the container runtime by using the nvidia-ctk command:

nvidia-ctk runtime configure --runtime=docker

pip-autoremove torch 
'''
