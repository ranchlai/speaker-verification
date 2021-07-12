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
import subprocess
import time
import warnings

import h5py
import numpy as np
import paddle
import paddleaudio
import sox
import yaml
#from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.utils import download
from paddleaudio.utils import augments
from paddleaudio.utils.logging import get_logger

logger = get_logger(__file__)


def random_choice(a):
    i = np.random.randint(0, high=len(a))
    return a[int(i)]


def read_list(file):
    lines = open(file).read().split('\n')
    keys = [
        '-'.join(l.split('/')[-3:]).split('.')[0] for l in lines
        if l.startswith('/')
    ]
    speakers = [k.split('-')[0] for k in keys]
    files = [l for l in lines if l.startswith('/')]
    return keys, speakers, files


def read_scp(file):
    lines = open(file).read().split('\n')
    keys = [l.split()[0] for l in lines if l.startswith('id')]
    speakers = [l.split()[0].split('-')[0] for l in lines if l.startswith('id')]
    files = [l.split()[1] for l in lines if l.startswith('id')]
    return keys, speakers, files


def augment_by_sox(wav, sr):
    tfm = sox.Transformer()
    effects = ['reverb', 'fade']  #'echo']'compand',
    num_effets = random.randint(1, len(effects))
    for e in random.sample(effects, num_effets):
        tfm.__getattribute__(e)()
    wav = tfm.build_array(input_array=wav, sample_rate_in=sr)
    return wav


from collections import defaultdict


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
                 speaker_set=None,
                 augment_with_sox=True,
                 augment_prob=0.2,
                 training=True,
                 n_uttns=8):
        #):
        super(Dataset, self).__init__()
        self.keys, self.speakers, self.files = read_list(scp)
        self.spk2files = defaultdict(list)
        for s, f in zip(self.speakers, self.files):
            self.spk2files[s].append(f)
        self.key2file = {k: f for k, f in zip(self.keys, self.files)}
        self.n_files = len(self.files)
        if speaker_set:
            if isinstance(speaker_set, str):
                with open(speaker_set) as f:
                    self.speaker_set = f.read().split('\n')
                    print(self.speaker_set[:10])
            else:
                self.speaker_set = speaker_set
        else:
            self.speaker_set = list(set(self.speakers))
            self.speaker_set.sort()
            # with open('../data/spaker_set.txt','wt') as f:
            #     f.write('\n'.join(self.speaker_set))
        self.spk2cls = {s: i for i, s in enumerate(self.speaker_set)}
        self.n_class = len(self.speaker_set)
        logger.info(f'speaker size: {self.n_class}')
        logger.info(f'file size: {self.n_files}')
        self.augment = augment
        self.augment_prob = augment_prob
        self.augment_with_sox = augment_with_sox
        self.training = training
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_uttns = n_uttns
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

    def get_wav(self, file, spk_id):
        file_duration = None
        if not self.augment and self.duration:
            file_duration = self.duration
        while True:
            try:
                wav, sr = paddleaudio.load(file,
                                           sr=self.sample_rate,
                                           duration=file_duration)
                break
            except:  #try another file
                logger.info(f'error loading file {file}')
                file = random_choice(self.spk2files[spk_id])
                time.sleep(1.0)
        if self.augment:
            wav = augments.random_crop_or_pad1d(wav, self.duration)
        elif self.duration:
            wav = augments.center_crop_or_pad1d(wav, self.duration)

        if self.augment_with_sox and random.random(
        ) < self.augment_prob:  #sox augment
            wav = augment_by_sox(wav, sr)

        return wav

    def __getitem__(self, idx):
        #choose a speaker
        spk_id = self.speaker_set[np.random.randint(self.n_class)]
        #print('idx:',idx)
        #spk_id = self.speaker_set[idx % self.n_class]
        # print('spk_id',spk_id)
        wavs = []
        for i in range(self.n_uttns):
            file = random_choice(self.spk2files[spk_id])
            wav = self.get_wav(file, spk_id)
            wavs += [wav[None, :]]

        wavs = np.concatenate(wavs, 0)
        return wavs,

    def __len__(self):
        return len(self.keys) // self.n_uttns


def worker_init(worker_id):
    time.sleep(worker_id / 32)
    np.random.seed(int(time.time()) % 100 + worker_id)


def get_train_loader(config):
    dataset = Dataset(config['spk_scp'],
                      keys=config['train_keys'],
                      speaker_set=config['speaker_set'],
                      augment=True,
                      augment_with_sox=config['augment_with_sox'],
                      duration=config['duration'],
                      n_uttns=config['n_uttns'])
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
                      speaker_set=config['speaker_set'],
                      augment=False,
                      augment_with_sox=False,
                      duration=config['duration'],
                      n_uttns=config['n_uttns'])
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
    #val_loader = get_val_loader(config)
    for i, (x, ) in enumerate(train_loader()):
        print(x.shape)
        break
