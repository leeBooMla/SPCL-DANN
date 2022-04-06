# SPCL-DANN
### Installation

```shell
git clone https://github.com/leeBooMla/SPCL-DANN

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset#data-for-visda2020-chanllenge)
```
SPCL-DANN/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
```

### Prepare ImageNet Pre-trained Models for IBN-Net
When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `logs/pretrained/`.
```shell
mkdir logs && cd logs
mkdir pretrained
```
The file tree should be
```
SPCL-DANN/logs
└── pretrained
    └── resnet50_ibn_a.pth.tar
```
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.


## Training

We utilize 4 GTX-280TI GPUs for training. **Note that**

+ The training is end-to-end, which means that no source-domain pre-training is required.
+ use `--iters 400` (default) for Market-1501 and PersonX datasets
+ use `--width 128 --height 256` (default) for person datasets
+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

### SPCL
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/spcl_train_uda.py \
  -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_OF_LOGS
```

**Some examples:**
```shell
### PersonX -> Market-1501 ###
# use all default settings is ok
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/spcl_train_uda.py \
  -ds personx -dt market1501 --logs-dir logs/spcl_uda/personx2market_resnet50


### SPCL+DANN
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/spcl_dann_train_uda.py \
  -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_OF_LOGS
```

**Some examples:**
```shell
### PersonX -> Market-1501 ###
# use all default settings is ok
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/spcl_dann_train_uda.py \
  -ds personx -dt market1501 --logs-dir logs/spcl_uda/personx2market_resnet50
```


## Evaluation

We utilize 1 GTX-2080TI GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets;
+ use `--dsbn` for domain adaptive models, and add `--test-source` if you want to test on the source domain;
+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

### Unsupervised Domain Adaptation

To evaluate the domain adaptive model on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py --dsbn \
  -d $DATASET --resume $PATH_OF_MODEL
```

To evaluate the domain adaptive model on the **source-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py --dsbn --test-source \
  -d $DATASET --resume $PATH_OF_MODEL
```

**Some examples:**
```shell
### PersonX -> Market1501 ###
# test on the target domain
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py --dsbn \
  -d market1501 --resume logs/spcl_uda/personx2market_resnet50/model_best.pth.tar
# test on the source domain
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py --dsbn --test-source \
  -d personx --resume logs/spcl_uda/personx2market_resnet50/model_best.pth.tar
```


