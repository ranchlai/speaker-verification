# Speaker verification using and ResnetSE ECAPA-TDNN
## Introduction
In this example, we demonstrate how to use PaddleAudio to train two types of networks for speaker verification.
The networks we support here are
- Resnet34 with Squeeze-and-excite block \[1\] to adaptively re-weight the feature maps.
- ECAPA-TDNN  \[2\]


## Datasets
### Training datasets
Following from this example and this example, we use the dev split [VoxCeleb 1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) which consists aof `1,211` speakers and the dev split of [VoxCeleb 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) consisting of `5,994` speakers for training. Thus there are `7,502` speakers totally in our training set.

Please download the two datasets from the [official website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) and unzip all audio into a folder, e.g., `./data/voxceleb/`. Make sure there are `7502` subfolders with prefix  `id1****` under the folder. You don't need to further process the data because all data processing such as adding noise / reverberation / speed perturbation  will be done on-the-fly. However, to speed up audio decoding, you can manually convert the m4a file in VoxCeleb 2 to wav file format, at the expanse of using more storage.

Finally, create a txt file that contains the list of audios for training by
```
cd ./data/voxceleb/
find `pwd`/ --type f > vox_files.txt
```
## Testing datasets
The testing split of VoxCeleb 1 is used for measuring the performance of speaker verification duration training and after the training completes.  You will need to download the data and unzip into a folder, e.g, `./data/voxceleb/test/`.

Then download the text files which list utterance  pairs to compare and the true labels indicating whether the utterances come from the same speaker. There are multiple trials and we will use [veri_test2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt).

## Training
To train your model from scratch, first create a folder(workspace) by

```
mkdir run1
cd run1

```
Copy an example config to your workspace by
```
cp ../eg/config.yaml .
```
Then change the config file accordingly to make sure all audio files can be correctly located(including the files used for data augmentation). Also you can change the training and model hyper-parameters  to suit your need.

Finally start your training by
```sh
python ../train.py -c ../run1/config.yaml  -d gpu:0
```



## Testing
To compute the eer after training completes, run
```
cd run1
python ../test.py -w <checkpoint_path> -c config.yaml  -d gpu:0
```


## Results

We compare our results  with [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

### Pretrained model of voxceleb_trainer

The test list is veri_test2.txt, which can be download from here [VoxCeleb1 (cleaned)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt)

| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|400|1.06%|
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|all|1.18%|

### This repo
| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + SAP + CMSoftmax| [tba] | [tba]|all|0.93%|
| ECAPA-TDNN + AAMSoftmax | [tba] | [tba] |all|1.10%|

## Reference

- [1] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141
- [2] Desplanques B, Thienpondt J, Demuynck K. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification[J]. arXiv preprint arXiv:2005.07143, 2020.