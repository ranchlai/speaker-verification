# Speaker verification using ECAPA_TDNN and ResnetSE



## Compare with [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer)

### Pretrained model of voxceleb_trainer

The test list is veri_test2.txt, which can be download from here [VoxCeleb1 (cleaned)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt)

| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|400|1.06%|
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|all|1.18%|

### This repo
| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + SAP + AMSoftmax| - | |all|1.14%|
| ECAPA-TDNN + AAMSoftmax | - | |all|1.19%|
