# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import pickle
import random
from typing import Any, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleaudio
import paddleaudio.functional as F
from paddle import Tensor
from paddle.utils import download

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'MixUpLoss', 'mixup_data',
    'get_txt_from_url', 'get_feature_from_url'
]


def get_txt_from_url(url):
    """Download and read text lines from url, remove empty lines if any.
    """
    file_path = download.get_weights_path_from_url(url)
    with open(file_path) as f:
        lines = f.read().split('\n')
    return [l for l in lines if len(l) > 0]


def get_pickle_results(url):
    weight = download.get_weights_path_from_url(url)
    with open(weight, 'rb') as f:
        results = pickle.load(f)
    return results


def get_feature_from_url(url):
    """Download and read features as numpy array from url.
    """
    file_path = download.get_weights_path_from_url(url)
    feature = np.load(file_path)
    return feature


def save_checkpoint(model_dir, step, model, optimizer, loss, prefix):
    print(f'checkpointing at step {step}')
    paddle.save(model.state_dict(),
                model_dir + '/{}_checkpoint{}.pdparams'.format(prefix, step))
    paddle.save(optimizer.state_dict(),
                model_dir + '/{}_checkpoint{}.pdopt'.format(prefix, step))


def load_checkpoint(model_dir, epoch, prefix):
    file = model_dir + '/{}_checkpoint_model{}.tar'.format(prefix, epoch)
    print('loading checkpoing ' + file)
    model_dict = paddle.load(
        model_dir + '/{}_checkpoint_epoch{}.pdparams'.format(prefix, epoch))
    optim_dict = paddle.load(
        model_dir + '/{}_checkpoint_epoch{}.pdopt'.format(prefix, epoch))
    return model_dict, optim_dict


class MixUpLoss(paddle.nn.Layer):
    """Define the mixup loss used in training audioset.

    Reference:
    Zhang, Hongyi, et al. “Mixup: Beyond Empirical Risk Minimization.” International Conference on Learning Representations, 2017.
    """
    def __init__(self, criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, mixup_target):
        assert type(mixup_target) in [
            tuple, list
        ] and len(mixup_target
                  ) == 3, 'mixup data should be tuple consists of (ya,yb,lamda)'
        ya, yb, lamda = mixup_target
        return lamda * self.criterion(pred, ya) \
                + (1 - lamda) * self.criterion(pred, yb)

    def extra_repr(self):
        return 'MixUpLoss with {}'.format(self.criterion)


def mixup_data(x, y, p=None, alpha=1.0):
    """Mix the input data and label using mixup strategy,  returns mixed inputs,
    pairs of targets, and lambda

    Reference:
    Zhang, Hongyi, et al. “Mixup: Beyond Empirical Risk Minimization.” International Conference on Learning Representations, 2017.

    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * paddle.index_select(x, index)
    y_a, y_b = y, paddle.index_select(y, index)
    mixed_target = (y_a, y_b, lam)
    if p is not None:
        mixed_p = lam * p + (1 - lam) * paddle.index_select(p, index)
        return mixed_x, mixed_p, mixed_target
    else:
        return mixed_x, None, mixed_target


class NoiseSource:
    """Read audio files randomly or sequentially from disk and pack them as a tensor.
    Parameters:
        audio_path_or_files(os.PathLike|List[os.PathLike]]): the audio folder or the audio file list.
        sample_rate(int): the target audio sample rate. If it is different from the native sample rate,
            resampling method will be invoked.
        duration(float): the duration after the audio is loaded. Padding or random cropping will take place
            depending on whether actual audio length is shorter or longer than int(sample_rate*duration).
            The audio tensor will have shape [batch_size, int(sample_rate*duration)]
        batch_size(int): the number of audio files contained in the returned tensor.
        random(bool): whether to read audio file randomly. If False, will read them sequentially.
            Default: True.
    Notes:
        In sequential mode, once the end of audio list is reached, the reader will start over again.
        The AudioSource object can be called endlessly.
     Shapes:
        - output: 2-D tensor with shape [batch_size, int(sample_rate*duration)]
    Examples:
        .. code-block:: python
        import paddle
        import paddleaudio.transforms as T
        reader = AudioSource(<audio_folder>, sample_rate=16000, duration=3.0, batch_size=2)
        audio = reader(x)
        print(audio.shape)
        >> [2,48000]
    """
    def __init__(self,
                 audio_path_or_files: Union[os.PathLike, List[os.PathLike]],
                 sample_rate: int,
                 duration: float,
                 batch_size: int,
                 random: bool = True):
        if isinstance(audio_path_or_files, list):
            self.audio_files = audio_path_or_files
        elif os.path.isdir(audio_path_or_files):
            self.audio_files = glob.glob(audio_path_or_files + '/*.wav',
                                         recursive=True)
            if len(self.audio_files) == 0:
                raise FileNotFoundError(
                    f'no files were found in {audio_path_or_files}')
        elif os.path.isfile(audio_path_or_files):
            self.audio_files = [audio_path_or_files]
        else:
            raise ValueError(
                f'rir_path_or_files={audio_path_or_files} is invalid')

        self.n_files = len(self.audio_files)
        self.idx = 0
        self.random = random
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration = int(duration * sample_rate)
        self._data = paddle.zeros((self.batch_size, self.duration),
                                  dtype='float32')

    def load_wav(self, file: os.PathLike):
        s, _ = paddleaudio.load(file, sr=self.sample_rate)
        s = paddle.to_tensor(s)
        s = F.random_cropping(s, target_size=self.duration)
        s = F.center_padding(s, target_size=self.duration)

        return s

    def __call__(self) -> Tensor:

        if self.random:
            files = [
                random.choice(self.audio_files) for _ in range(self.batch_size)
            ]
        else:
            files = []
            for _ in range(self.batch_size):
                file = self.audio_files[self.idx]
                self.idx += 1
                if self.idx >= self.n_files:
                    self.idx = 0
                files += [file]
        for i, f in enumerate(files):
            self._data[i, :] = self.load_wav(f)

        return self._data

    def __repr__(self):
        return (
            self.__class__.__name__ +
            f'(n_files={self.n_files}, random={self.random}, sample_rate={self.sample_rate})'
        )


class RIRSource(nn.Layer):
    """Gererate RIR filter coefficients from local file sources.
    Parameters:
        rir_path_or_files(os.PathLike|List[os.PathLike]): the directory that contains rir files directly
        (without subfolders) or the list of rir files.
    Examples:
        .. code-block:: python
        import paddle
        import paddleaudio.transforms as T
        reader = T.RIRSource(<rir_folder>, sample_rate=16000, random=True)
        weight = reader()
    """
    def __init__(self,
                 rir_path_or_files: Union[os.PathLike, List[os.PathLike]],
                 sample_rate: int,
                 random: bool = True):
        super(RIRSource, self).__init__()
        if isinstance(rir_path_or_files, list):
            self.rir_files = rir_path_or_files
        elif os.path.isdir(rir_path_or_files):
            self.rir_files = glob.glob(rir_path_or_files + '/*.wav',
                                       recursive=True)
            if len(self.rir_files) == 0:
                raise FileNotFoundError(
                    f'no files were found in {rir_path_or_files}')
        elif os.path.isfile(rir_path_or_files):
            self.rir_files = [rir_path_or_files]
        else:
            raise ValueError(
                f'rir_path_or_files={rir_path_or_files} is invalid')

        self.n_files = len(self.rir_files)
        self.idx = 0
        self.random = random
        self.sample_rate = sample_rate

    def forward(self) -> Tensor:
        if self.random:
            file = random.choice(self.rir_files)
        else:
            i = self.idx % self.n_files
            file = self.rir_files[i]
            self.idx += 1
            if self.idx >= self.n_files:
                self.idx = 0

        rir, _ = paddleaudio.load(file, sr=self.sample_rate, mono=True)
        rir_weight = paddle.to_tensor(rir[None, None, ::-1])
        rir_weight = paddle.nn.functional.normalize(rir_weight, p=2, axis=-1)
        return rir_weight

    def __repr__(self):
        return (
            self.__class__.__name__ +
            f'(n_files={self.n_files}, random={self.random}, sample_rate={self.sample_rate})'
        )
