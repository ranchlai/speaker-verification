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
import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleaudio as pa
import yaml
from dataset import get_val_loader
from model import resnet50
from paddle.utils import download
from paddleaudio.utils.log import logger
from models import ResNetSE34

checkpoint_url = ''


def evaluate(model, test_list_file, test_folder):
    model.eval()
    avg_acc = 0.0
    for batch_id, (x,y) in enumerate(val_loader()):
        x = x.unsqueeze((1))
        label = y
        logits = model(x)
        pred = F.log_softmax(logits)
        acc = np.mean(np.argmax(pred.numpy(), axis=1) == y.numpy())
        avg_acc = (avg_acc * batch_id + acc) / (1 + batch_id)
        msg = f'eval epoch:{epoch}, batch:{batch_id}'
        msg += f'|{len(val_loader)}'
        msg += f',acc:{avg_acc:.3}'

        if batch_id % 100 == 0:
            logger.info(msg)

    return avg_acc

def get_feature(file,model,config):
    wav,_ = pa.load(file,sr=config['fbank']['sr'])
    x = pa.melspectrogram(wav,**config['fbank'])
    x = paddle.to_tensor(x)
    x = x.unsqueeze((0,1))
    with paddle.no_grad():
        _,feature = model(x)
    return feature

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=False, default='config.yaml')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")

    parser.add_argument('--weight', type=str, required=False, default='')
    args = parser.parse_args([])
    args.config = './eg1/config.yaml'
    args.weight = './eg1/checkpoints/tdnn_amsoftmax_epoch13_acc0.969.pdparams'
    args.test_list = './data/veri_test.txt'
    args.test_folder = './data/voxceleb1'


   
    with open(args.config) as f:
        config = yaml.safe_load(f)
    paddle.set_device(args.device)
    logger.info(f'using ' + config['model']['name'])
    ModelClass = eval(config['model']['name'])
    model = ModelClass(**config['model']['params'])
    if args.weight.strip() == '':
        logger.info(
            f'Using pretrained weight: {checkpoint_url}')
        args.weight = download.get_weights_path_from_url(checkpoint_url[
            args.task_type])

    model.load_dict(paddle.load(args.weight))
    model.eval()


    with open(args.test_list) as f:
        lines = f.read().split('\n')
    label_wav_pairs = [l.split() for l in lines if len(l)>0]
    logger.info(f'{len(label_wav_pairs)} test pairs listed')

    labels = []
    scores = []
    for label, f1, f2 in label_wav_pairs:
        full_path1 = os.path.join(args.test_folder,f1)
        full_path2 = os.path.join(args.test_folder,f2)
        feature1 = get_feature(full_path1,model,config)
        feature2 = get_feature(full_path2,model,config)
        score = float(paddle.dot(feature1.squeeze(),feature2.squeeze()))
        labels.append(label)
        scores.append(score)

        
    

   # val_loader = get_val_loader(c)

    logger.info(f'Evaluating...')
    val_loss, val_acc = evaluate(
        0, val_loader, model, nn.NLLLoss(), task_type=args.task_type)
    logger.info(f'Overall acc: {val_acc:.3}')
    logger.info(f'Overall loss: {val_loss:.3}')
