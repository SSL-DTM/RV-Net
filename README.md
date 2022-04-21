# Relief Visualization Network (RV-Net)

This repository is for the encoder-decoder pretext for the paper titled "Self Supervised Learning for Semantic Segmentation of Archaeological 
Monuments in DTMs".

The code is mainly copied and adapted from the original [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
implementation. 

Create your own dataset module under `encdec/data` by inheriting the `BaseDataset` class from 
`encdec/data/base_dataset.py` and customize it for your own dataset.

The default configuration file for training/testing is in `encdec/config/default.py`. Customize/create
your own specific configuration in `configs/hrnet_config.yaml`. The `hrnet` part in the name is
mandatory so that the model history/files will be saved properly.

The code works with `Python 3.7.10` and `PyTorch 1.8.1`. 

Follow these steps:

- Clone this repository
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) and run the following
command to create the environment and install necessary libraries:

- `conda env create -f environment.yml`

This should create an environment called `ikg`. Activate the environment and install the package locally to train/test a model
on your own dataset:

- `conda activate ikg`
- `pip install -e .`

To train a model using 4 GPUS, run:
- `python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg configs/hrnet_config.yaml`

To train a model using 2 GPUS, run:
- `python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg configs/hrnet_config.yaml GPUS 0,1`

To test a model, run:
- `python tools/test.py --cfg configs/hrnet_config.yaml TEST.MODEL_FILE PATH/TO/SAVED/MODEL/best.pth GPUS 0`


The pretrained model weights for RV-Net are available [here](https://github.com/SSL-DTM/model_weights/releases/download/v0.0.0/RVNet.pth). It can be used for initializing and finetuning semantic segmentation model. Check out the [semantic segmentation](https://github.com/SSL-DTM/semantic_segmentation) repo for how to do this.



