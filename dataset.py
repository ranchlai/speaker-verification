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
from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.utils import download
from paddleaudio import augment
from paddleaudio.utils.log import logger



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


def get_keys(file_pointers):
    all_keys = []
    key2file = {}
    for fp in file_pointers:
        try:
            keys = list(fp.keys())
            all_keys += keys
            key2file.update({k: fp for k in keys})
        except:
            logger.info(f'failed to load {fp}')
        
    return all_keys, key2file

def read_spk_scp(file):
    lines = open(file).read().split('\n')
    spk2id = {l.split()[0]:int(l.split()[1]) for l in lines if l.startswith('id')}
    return spk2id
class H5AudioSet(Dataset):
    """
    Dataset class for Audioset, with mel features stored in multiple hdf5 files.
    The h5 files store mel-spectrogram features pre-extracted from wav files.
    Use wav2mel.py to do feature extraction.
    """

    def __init__(self,
                 h5_files,
                 config,
                 keys = None,
                 augment=True,
                 training=True,
                 balanced_sampling=True):
        super(H5AudioSet, self).__init__()
        self.h5_files = h5_files
        self.config = config
        self.file_pointers = [h5py.File(f) for f in h5_files]
        self.all_keys, self.key2file = get_keys(self.file_pointers)
        if keys is not None:
            self.all_keys = list(set.intersection(set(self.all_keys),set(keys)))
        n_class = len(set([k.split('-')[0] for k in self.all_keys]))
        logger.info(f'Totally {len(self.all_keys)} keys and {n_class} classes listed')
        self.augment = augment
        self.training = training
        self.balanced_sampling = balanced_sampling
        logger.info(
            f'{len(self.h5_files)} h5 files, totally {len(self.all_keys)} audio files listed'
        )
        self.key2clsidx = read_spk_scp(self.config['spk_scp'])
        self.clsidx2key = {self.key2clsidx[k]:k for k in self.key2clsidx} 

    def _process(self, x):
        assert x.shape[0] == self.config[
            'mel_bins'], 'the first dimension must be mel frequency'

        target_len = self.config['max_mel_len']
        if x.shape[1] <= target_len:
            pad_width = (target_len - x.shape[1]) // 2 + 1
            x = np.pad(x, ((0, 0), (pad_width, pad_width)))
        x = x[:, :target_len]

        if self.training and self.augment:
            x = augment.random_crop2d(
                x, self.config['mel_crop_len'], tempo_axis=1)
            x = spect_permute(x, tempo_axis=1, nblocks=random_choice([0, 2, 3]))
            aug_level = random_choice([0.2, 0.1, 0])
            x = augment.adaptive_spect_augment(x, tempo_axis=1, level=aug_level)
        #x = x[:,:self.config['mel_crop_len']] # should be center-crop
        return x

    def __getitem__(self, idx):

        if self.balanced_sampling:
            cls_id = int(np.random.randint(0, self.config['num_classes']))
            keys = self.clsidx2key[cls_id]
            k = random_choice(keys)
            cls_ids = self.key2clsidx[k]
        else:
            idx = idx % len(self.all_keys)
            k = self.all_keys[idx]
            cls_ids = self.key2clsidx[k]
        fp = self.key2file[k]
        x = fp[k][:, :]
        x = self._process(x)
        #prob = np.array(file2feature[k], 'float32')

        #y = np.zeros((self.config['num_classes'], ), 'float32')
        #y[cls_ids] = 1.0
        return x, cls_ids

    def __len__(self):
        return len(self.all_keys)


def worker_init(worker_id):
    time.sleep(worker_id / 32)
    np.random.seed(int(time.time()) % 100 + worker_id)


def get_train_loader(config):
    with open(config['train_keys']) as f:
        train_keys = f.read().split('\n')

    h5_files = glob.glob(config['data'])
    train_dataset = H5AudioSet(
        h5_files,
        config,
        keys = train_keys,
        balanced_sampling=config['balanced_sampling'],
        augment=True,
        training=True)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        num_workers=config['num_workers'],
        use_buffer_reader=True,
        use_shared_memory=True,
        worker_init_fn=worker_init)

    return train_loader


def get_val_loader(config):

    with open(config['val_keys']) as f:
        val_keys = f.read().split('\n')
        
    h5_files = glob.glob(config['data'])
    val_dataset = H5AudioSet(
        h5_files,
         config,
         keys = val_keys, 
        balanced_sampling=False,
         augment=False)
    val_loader = DataLoader(
        val_dataset,
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
    import pdb;pdb.set_trace()
    print(train_loader.dataset[0])
   # for i,(x,y) in enumerate(train_loader):
       # print(x.shape,y)
  
    # dataset = H5AudioSet(
    #     [config['eval_h5']], config, balanced_sampling=False, augment=False)
    # x, y, p = dataset[0]
    # logger.info(f'{x.shape}, {y, p.shape}')
