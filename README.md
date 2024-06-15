# -CVPRW-CSCO
This repository is the official implementation of "CSCO: Connectivity Search of Convolutional Operators".

## Requirements
There are a few action lists before using this repository:
- Please refer to ``environment.yaml'' to install the basic requirements needed for this project. 
- Please download CIFAR-10 dataset following [this](https://www.cs.toronto.edu/~kriz/cifar.html).
- Please download ImageNet dataset following [this](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).

## Overview
This repository is divided into two parts: `search` and `evaluation`. Please cd to each directory before proceeding.


## Evaluating the best models
To evaluate the best model on ImageNet, you may do the following:
```
cd evaluation/pytorch-image-models-main
./scripts/train_csco_base.sh
```

Note that the implementation is based on PyTorch ImageNet [TIMM](https://github.com/huggingface/pytorch-image-models). You may checkout the latest version if you want to apply updates and utilize the latest technology.


## Reference
If you want to use this project, please kindly cite our research paper:
```
@article{zhang2024csco,
  title={CSCO: Connectivity Search of Convolutional Operators},
  author={Zhang, Tunhou and Li, Shiyu and Cheng, Hsin-Pai and Yan, Feng and Li, Hai and Chen, Yiran},
  journal={arXiv preprint arXiv:2404.17152},
  year={2024}
}
```

## Acknowledgement
This project is partly supported by NSF 2112562, ARO W911NF-23-2-0224 and NSF CAREER-2305491. We thank [timm](https://github.com/huggingface/pytorch-image-models) for providing implementation for ImageNet training recipe.