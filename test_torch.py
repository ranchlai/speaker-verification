import torch
import torchaudio
import yaml
import librosa
import numpy as np
from models_torch import ResNetSE34V2

from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import amplitude_to_DB


def load_audio(file):
    EPS = 1e-8
    s, _ = librosa.load(file, sr=16000)
    amax = np.max(np.abs(s))
    factor = 1.0 / (amax + EPS)
    s = s * factor
    return s


with open('egs/resnet/config.yaml') as f:
    config = yaml.safe_load(f)

sd = torch.load('./resnetse34_epoch92_eer0.00931.pth')
model = ResNetSE34V2(nOut=256, n_mels=config['fbank']['n_mels'])
model.load_state_dict(sd)
model.eval()
torch.set_grad_enabled(False)

transform = MelSpectrogram(
    sample_rate=config['fbank']['sr'],
    n_fft=config['fbank']['n_fft'],
    win_length=config['fbank']['win_length'],
    hop_length=config['fbank']['hop_length'],
    window_fn=torch.hamming_window,
    n_mels=config['fbank']['n_mels'],
    f_min=config['fbank']['f_min'],
    f_max=config['fbank']['f_max'],
    norm='slaney')

s = load_audio('./data/test_audio.wav')
x = torch.tensor(s[None, :])
x = transform(x)
x = amplitude_to_DB(
    x, multiplier=10, amin=config['fbank']['amin'], db_multiplier=0, top_db=75)

feature = model(x[:, None, :, :])
feature = torch.nn.functional.normalize(feature)
