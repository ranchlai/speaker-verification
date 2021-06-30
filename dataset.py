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
import subprocess
import time
import warnings

import h5py
import numpy as np
import paddle
import paddleaudio
import yaml
#from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.utils import download
from paddleaudio import augment
from paddleaudio.utils.logging import get_logger

logger = get_logger(__file__)


def spect_permute(spect, tempo_axis, nblocks):
    """spectrogram  permutaion"""
    assert spect.ndim == 2, 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape
    if nblocks <= 1:
        return spect

    block_width = nt // nblocks + 1
    if tempo_axis == 1:
        blocks = [
            spect[:, block_width * i:(i + 1) * block_width]
            for i in range(nblocks)
        ]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks, 1)
    else:
        blocks = [
            spect[block_width * i:(i + 1) * block_width, :]
            for i in range(nblocks)
        ]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks, 0)
    return new_spect


def random_choice(a):
    i = np.random.randint(0, high=len(a))
    return a[int(i)]


def read_scp(file):
    lines = open(file).read().split('\n')
    keys = [l.split()[0] for l in lines if l.startswith('id')]
    speakers = [l.split()[0].split('-')[0] for l in lines if l.startswith('id')]
    files = [l.split()[1] for l in lines if l.startswith('id')]
    return keys, speakers, files


class Dataset(paddle.io.Dataset):
    """
    Dataset class for Audioset, with mel features stored in multiple hdf5 files.
    The h5 files store mel-spectrogram features pre-extracted from wav files.
    Use wav2mel.py to do feature extraction.
    """
    def __init__(self,
                 scp,
                 keys=None,
                 sample_rate=16000,
                 duration=None,
                 augment=True,
                 training=True):
        #balanced_sampling=False):
        super(Dataset, self).__init__()

        self.keys, self.speakers, self.files = read_scp(scp)
        self.key2file = {k: f for k, f in zip(self.keys, self.files)}
        self.n_files = len(self.files)
        self.speaker_set = list(set(self.speakers))
        self.speaker_set.sort()
        self.spk2cls = {s: i for i, s in enumerate(self.speaker_set)}
        self.n_class = len(self.speaker_set)
        logger.info(f'speaker size: {self.n_class}')
        logger.info(f'file size: {self.n_files}')
        self.augment = augment
        self.training = training
        self.sample_rate = sample_rate
        #self.balanced_sampling = balanced_sampling
        self.duration = duration
        if augment:
            assert duration, 'if augment is True, duration must not be None'

        if self.duration:
            self.duration = int(self.sample_rate * self.duration)

        if isinstance(keys, list):
            self.keys = keys
        elif isinstance(keys, str):
            with open(keys) as f:
                self.keys = f.read().split('\n')
                self.keys = [k for k in self.keys if k.startswith('id')]

        logger.info(f'using {len(self.keys)} keys')

    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        key = self.keys[idx]
        spk = key.split('-')[0]
        #spk = self.speakers[idx]
        cls_idx = self.spk2cls[spk]
        file = self.key2file[key]

        #file = self.files[idx]
        file_duration = None
        if not self.augment and self.duration:
            file_duration = self.duration

        while True:
            try:
                wav, _ = paddleaudio.load(file,
                                          sr=self.sample_rate,
                                          duration=file_duration)
                break
            except:
                key = self.keys[idx]
                spk = key.split('-')[0]
                #spk = self.speakers[idx]
                cls_idx = self.spk2cls[spk]
                file = self.key2file[key]
        if self.augment:
            wav = paddleaudio.features.random_crop_or_pad1d(wav, self.duration)
        elif self.duration:
            wav = paddleaudio.features.center_crop_or_pad1d(wav, self.duration)

        return wav, cls_idx

    def __len__(self):
        return len(self.keys)


def worker_init(worker_id):
    time.sleep(worker_id / 32)
    np.random.seed(int(time.time()) % 100 + worker_id)


def get_train_loader(config):
    dataset = Dataset(config['spk_scp'],
                      keys=config['train_keys'],
                      augment=True,
                      duration=config['duration'])
    train_loader = paddle.io.DataLoader(dataset,
                                        shuffle=True,
                                        batch_size=config['batch_size'],
                                        drop_last=True,
                                        num_workers=config['num_workers'],
                                        use_buffer_reader=True,
                                        use_shared_memory=True,
                                        worker_init_fn=worker_init)

    return train_loader


def get_val_loader(config):

    dataset = Dataset(config['spk_scp'],
                      keys=config['val_keys'],
                      augment=False,
                      duration=config['duration'])
    val_loader = paddle.io.DataLoader(dataset,
                                      shuffle=False,
                                      batch_size=config['val_batch_size'],
                                      drop_last=False,
                                      num_workers=config['num_workers'])

    return val_loader


if __name__ == '__main__':
    # do some testing here
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    train_loader = get_train_loader(config)
    val_loader = get_val_loader(config)
    for i, (x, y) in enumerate(train_loader()):
        print(x, y)
        break

    for i, (x, y) in enumerate(val_loader()):
        print(x, y)
        break
