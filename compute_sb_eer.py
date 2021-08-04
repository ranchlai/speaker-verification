import argparse
import os

import numpy as np
#from paddleaudio.utils import get_logger
import torch
import torchaudio
import yaml
from speechbrain.pretrained import EncoderClassifier

import metrics

#logger = get_logger()

file2feature = {}


def get_feature(file, model):
    global file2feature
    if file in file2feature:
        return file2feature[file]

    signal, fs = torchaudio.load(file)
    embeddings = model.encode_batch(signal.cuda())

    #     s, _ = paddleaudio.load(file, sr=16000)
    #     s = paddle.to_tensor(s[None, :])
    #     s = melspectrogram(s).astype('float32')
    #     with paddle.no_grad():
    #         feature = model(s).squeeze()
    embeddings = torch.nn.functional.normalize(embeddings[0], p=2, dim=1)
    file2feature.update({file: embeddings})
    #import pdb;pdb.set_trace()
    return embeddings


def compute_eer(config, model):

    #     transforms = []
    #     melspectrogram = LogMelSpectrogram(**config['fbank'])
    #     transforms+=[melspectrogram]
    #     if config['normalize']:
    #         transforms +=[Normalize2(config['mean_std_file'])]

    #     transforms = Compose(transforms)

    global file2feature
    file2feature = {}
    test_list = config['test_list']
    test_folder = config['test_folder']
    with open(test_list) as f:
        lines = f.read().split('\n')
    label_wav_pairs = [l.split() for l in lines if len(l) > 0]
    print(f'{len(label_wav_pairs)} test pairs listed')
    labels = []
    scores = []
    for i, (label, f1, f2) in enumerate(label_wav_pairs):
        full_path1 = os.path.join(test_folder, f1)
        full_path2 = os.path.join(test_folder, f2)
        feature1 = get_feature(full_path1, model)
        feature2 = get_feature(full_path2, model)
        score = float(torch.dot(feature1.squeeze(), feature2.squeeze()))
        labels.append(label)
        scores.append(score)
        if i % (len(label_wav_pairs) // 10) == 0:
            print(f'processed {i}|{len(label_wav_pairs)}')

    scores = np.array(scores)
    labels = np.array([int(l) for l in labels])
    result = metrics.compute_eer(scores, labels)
    min_dcf = metrics.compute_min_dcf(result.fr, result.fa)
    print(f'eer={result.eer}, thresh={result.thresh}, minDCF={min_dcf}')
    return result, min_dcf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='config.yaml')
    parser.add_argument(
        '-d',
        '--device',
        #choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    # parser.add_argument('--test_list', type=str, required=True)
    # parser.add_argument('--test_folder', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})
    result, min_dcf = compute_eer(config, model)
    print(f'eer={result.eer}, thresh={result.thresh}, minDCF={min_dcf}')
