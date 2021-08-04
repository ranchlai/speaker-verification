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
from paddleaudio.transforms import Compose, RandomMasking, RandomMuCodec
from paddleaudio.utils.logging import get_logger

from balance_dataset import get_train_loader, get_val_loader
from evaluate import evaluate
from losses import AMSoftmaxLoss, ProtoTypical
from models import ResNetSE34, ResNetSE34V2
from utils import MixUpLoss, load_checkpoint, mixup_data, save_checkpoint

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
    parser.add_argument('--distributed', type=int, required=False, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config['log_dir'], exist_ok=True)
    logger = get_logger(__file__,
                        log_dir=config['log_dir'],
                        log_file_name='resnet_se34v2')

    prefix = f'tdnn_amsoftmax'

    if args.distributed != 0:
        dist.init_parallel_env()
        local_rank = dist.get_rank()
    else:
        paddle.set_device(args.device)
        local_rank = 0

    logger.info(f'using ' + config['model']['name'])
    ModelClass = eval(config['model']['name'])
    model = ModelClass(**config['model']['params'],
                       feature_config=config['fbank'])
    #define loss and lr
    LossClass = eval(config['loss']['name'])

    loss_fn = LossClass(**config['loss']['params'])
    #loss_fn = nn.NLLLoss()
    warm_steps = config['warm_steps']
    if warm_steps != 0:
        lrs = np.linspace(1e-10, config['start_lr'], warm_steps)

    params = model.parameters() + loss_fn.parameters()
    # params = model.parameters()
    # restore checkpoint

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

        augment_mel = Compose([freq_masking, time_masking])
        print(augment_mel)

    else:
        augment_mel = None
    if config['augment_wav']:
        augment_wav = RandomMuCodec()
        print(augment_wav)
    else:
        augment_wav = None

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=config['start_lr'],
        eta_min=1e-7,
        T_max=2000,
        verbose=False)

    if args.restore != -1:
        # model_dict, optim_dict = load_checkpoint(config['model_dir'],
        #args.restore, prefix)
        fn = os.path.join(config['model_dir'],
                          f'{prefix}_checkpoint_epoch{args.restore}.tar')
        ckpt = paddle.load(fn)
        model.load_dict(ckpt['model'])
        optimizer = Adam(learning_rate=scheduler, parameters=params)
        opti_state_dict = ckpt['opti']
        opti_state_dict.update(
            {'LR_Scheduler': {
                'last_epoch': 0,
                'last_lr': config['start_lr']
            }})
        optimizer.set_state_dict(opti_state_dict)
        loss_fn.load_dict(ckpt['loss'])

        start_epoch = args.restore + 1
    else:

        start_epoch = 0
        optimizer = Adam(learning_rate=scheduler, parameters=params)
    optimizer._learning_rate = 1e-3
    #
    os.makedirs(config['model_dir'], exist_ok=True)

    if args.distributed != 0:
        model = paddle.DataParallel(model)

    train_loader = get_train_loader(config)
    val_loader = get_val_loader(config)

    epoch_num = config['epoch_num']
    if args.restore != -1:
        eer = evaluate(start_epoch, val_loader, model, loss_fn)
        best_eer = eer
    else:
        best_eer = 1.0
    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epoch_num):
        avg_loss = 0.0
        avg_acc = 0.0
        model.train()
        model.clear_gradients()
        t0 = time.time()
        for batch_id, (x, ) in enumerate(train_loader()):
            if warm_steps != 0 and step < warm_steps:
                optimizer.set_lr(lrs[step])
            if step == warm_steps:
                logger.info(f'now using {scheduler}')
                optimizer._learning_rate = scheduler
            n_spk, n_uttn, _ = x.shape

            x = x.reshape((n_spk * n_uttn, x.shape[-1]))
            logits = model(x, augment_wav, augment_mel)
            logits = logits.reshape((n_spk, n_uttn, logits.shape[-1]))
            loss, pred = loss_fn(logits)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.clear_gradients()
            acc = np.mean(np.argmax(pred.numpy(), -1) == np.arange(n_spk))

            avg_loss = (avg_loss * batch_id + loss.numpy()[0]) / (1 + batch_id)
            avg_acc = (avg_acc * batch_id + acc) / (1 + batch_id)
            elapsed = (time.time() - t0) / 3600
            remain = elapsed / (1 + batch_id) * (len(train_loader) - batch_id)

            msg = f'epoch:{epoch}, batch:{batch_id}'
            msg += f'|{len(train_loader)}'
            msg += f', loss:{avg_loss:.3}'
            msg += f', acc:{avg_acc:.3}'
            msg += f', lr:{optimizer.get_lr():.2}'
            msg += f', elapsed:{elapsed:.3}h'
            msg += f', remained:{remain:.3}h'
            step += 1

            if step % config['log_step'] == 0 and local_rank == 0:
                logger.info(msg)

            if step % config['checkpoint_step'] == 0 and local_rank == 0:
                fn = os.path.join(config['model_dir'],
                                  f'{prefix}_checkpoint_epoch{epoch}.tar')

                obj = {
                    'model': model.state_dict(),
                    'loss': loss_fn.state_dict(),
                    'opti': optimizer.state_dict(),
                    'lr': optimizer.get_lr()
                }
                paddle.save(obj, fn)

            if step % config['eval_step'] == 0 and local_rank == 0:
                result, min_dcf = compute_eer(config, model,
                                              config['test_folder'])
                eer = result.eer
                model.train()
                model.clear_gradients()

                if eer < best_eer:
                    logger.info('eer improved from {} to {}'.format(
                        best_eer, eer))
                    best_eer = eer
                    fn = os.path.join(config['model_dir'],
                                      f'{prefix}_epoch{epoch}_eer{eer:.3}')
                    paddle.save(model.state_dict(), fn + '.pdparams')
                else:
                    logger.info(f'eer {eer} did not improved from {best_eer}')

            # if step % config['lr_dec_per_step'] == 0 and step != 0:
            #     if optimizer.get_lr() <= 1e-6:
            #         factor = 0.95
            #     else:
            #         factor = 0.8
            #     optimizer.set_lr(optimizer.get_lr() * factor)
            #     logger.info('decreased lr to {}'.format(optimizer.get_lr()))
