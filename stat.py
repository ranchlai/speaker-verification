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

import argparse
import glob
import os
import random
import time
from test import compute_eer

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import yaml
from paddle.optimizer import Adam
from paddle.utils import download
from paddleaudio.transforms import *
from paddleaudio.utils import get_logger

from dataset import get_train_loader
from evaluate import evaluate
from losses import AdditiveAngularMargin, AMSoftmaxLoss
from models import *
from utils import MixUpLoss, load_checkpoint, mixup_data, save_checkpoint


def get_lr(step, base_lr, max_lr, half_cycle=5000, reverse=False):
    if int(step / half_cycle) % 2 == 0:
        lr = (step % half_cycle) / half_cycle * (max_lr - base_lr)
        lr = base_lr + lr
    else:
        lr = (step % half_cycle / half_cycle) * (max_lr - base_lr)
        lr = max_lr - lr
    lr = max_lr - lr

    return lr


class Normalize:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        assert x.ndim == 3
        mean = paddle.mean(x, [1, 2], keepdim=True)
        std = paddle.std(x, [1, 2], keepdim=True)
        return (x - mean) / (std + self.eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audioset training')
    parser.add_argument(
        '-d',
        '--device',
        #choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument('-r', '--restore', type=int, required=False, default=-1)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-e',
                        '--eval_at_begin',
                        type=bool,
                        choices=[True, False],
                        required=False,
                        default=False)
    parser.add_argument('--distributed',
                        type=bool,
                        choices=[True, False],
                        required=False,
                        default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config['log_dir'], exist_ok=True)
    logger = get_logger(__file__,
                        log_dir=config['log_dir'],
                        log_file_name='resnet_se34v2')

    prefix = f'tdnn_amsoftmax'

    if args.distributed:
        dist.init_parallel_env()
        local_rank = dist.get_rank()
        print(local_rank)
    else:
        paddle.set_device(args.device)
        local_rank = 0

    logger.info(f'using ' + config['model']['name'])
    ModelClass = eval(config['model']['name'])
    model = ModelClass(**config['model']['params'])
    #define loss and lr
    LossClass = eval(config['loss']['name'])

    loss_fn = LossClass(**config['loss']['params'])
    loss_fn.train()
    #loss_fn = nn.NLLLoss()
    params = model.parameters() + loss_fn.parameters()

    transforms = []
    if config['augment_wav']:

        noise_source1 = NoiseSource(open(
            config['muse_speech']).read().split('\n')[:-1],
                                    sample_rate=16000,
                                    duration=config['duration'],
                                    batch_size=config['batch_size'])
        noisify1 = Noisify(noise_source1,
                           snr_high=config['muse_speech_srn_high'],
                           snr_low=config['muse_speech_srn_low'],
                           random=True)

        noise_source2 = NoiseSource(open(
            config['muse_music']).read().split('\n')[:-1],
                                    sample_rate=16000,
                                    duration=config['duration'],
                                    batch_size=config['batch_size'])
        noisify2 = Noisify(noise_source2,
                           snr_high=config['muse_music_srn_high'],
                           snr_low=config['muse_music_srn_low'],
                           random=True)
        noise_source3 = NoiseSource(open(
            config['muse_noise']).read().split('\n')[:-1],
                                    sample_rate=16000,
                                    duration=config['duration'],
                                    batch_size=config['batch_size'])
        noisify3 = Noisify(noise_source3,
                           snr_high=config['muse_noise_srn_high'],
                           snr_low=config['muse_noise_srn_low'],
                           random=True)
        rir_files = open(config['rir_path']).read().split('\n')[:-1]
        random_rir_reader = RIRSource(rir_files, random=True, sample_rate=16000)
        reverb = Reverberate(rir_source=random_rir_reader)
        muse_augment = RandomChoice([noisify1, noisify2, noisify3])
        wav_augments = RandomApply([muse_augment, reverb])
        transforms += [wav_augments]
    melspectrogram = LogMelSpectrogram(**config['fbank'])
    transforms += [melspectrogram]
    if config['normalize']:
        transforms += [Normalize()]

    if config['augment_mel']:
        #define spectrogram masking
        time_masking = RandomMasking(
            max_mask_count=config['max_time_mask'],
            max_mask_width=config['max_time_mask_width'],
            axis=-1)
        freq_masking = RandomMasking(
            max_mask_count=config['max_freq_mask'],
            max_mask_width=config['max_freq_mask_width'],
            axis=-2)

        mel_augments = RandomApply([freq_masking, time_masking], p=0.25)
        transforms += [mel_augments]
    transforms = Compose(transforms)
    print(transforms)

    if args.restore != -1:
        fn = os.path.join(config['model_dir'],
                          f'{prefix}_checkpoint_epoch{args.restore}.tar')
        ckpt = paddle.load(fn)
        model.load_dict(ckpt['model'])
        optimizer = Adam(learning_rate=config['max_lr'], parameters=params)
        opti_state_dict = ckpt['opti']
        optimizer.set_state_dict(opti_state_dict)
        loss_fn.load_dict(ckpt['loss'])
        start_epoch = args.restore + 1
    else:
        start_epoch = 0
        optimizer = Adam(learning_rate=config['max_lr'], parameters=params)
    os.makedirs(config['model_dir'], exist_ok=True)

    if args.distributed:
        model = paddle.DataParallel(model)
    train_loader = get_train_loader(config)
    epoch_num = config['epoch_num']
    print(args.eval_at_begin)
    if args.restore != -1 and local_rank == 0 and args.eval_at_begin:
        result, min_dcf = compute_eer(config, model)
        best_eer = result.eer  #0.022#result.eer
        logger.info(f'eer: {best_eer}')
    else:
        best_eer = 1.0
    step = start_epoch * len(train_loader)
    avg_mean = paddle.zeros((80, ))
    avg_std = paddle.zeros((80, ))

    for batch_id, (x, y) in enumerate(train_loader()):
        x = transforms(x)

        mean = paddle.mean(x, [0, 2])
        avg_mean = (avg_mean * batch_id + mean) / (batch_id + 1)
        std = paddle.mean(x, [0, 2])
        avg_std = (avg_std * batch_id + std) / (batch_id + 1)
        if batch_id % 100 == 0:
            logger.info(f'{batch_id}|{len(train_loader())}')

        if batch_id % 1000 == 0:
            logger.info(f'mean {avg_mean.numpy()}')
            logger.info(f'std {avg_std.numpy()}')

            paddle.save({'mean': avg_mean, 'std': avg_std}, 'stat.pd')
