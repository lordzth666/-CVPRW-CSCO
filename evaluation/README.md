# Evaluating the best network discovered by CSCO

This folder contains the evaluation protocol for CSCO. Please visit ``evaluation/scripts`` for the detailed evaluation protocol. We offer two types of models: base model as a product of our end-to-end search, and scaled models with the optimal performnace (provided by the paper.)

You should first cd to the project directory:
```sh
cd evaluation/pytorch-image-models-main
```

To run the best model (paper result):
```sh
./distributed_train.sh 4 /imagenet/ --model gram224 -b 192 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 3e-5 --drop 0.0 --drop-path 0.0 --model-ema --model-ema-decay 0.9999 --reprob 0.0 --amp --lr .048 --gram_proto gram_export/csco/csco_im_large.prototxt --data-dir ./data/ImageNet --bn-momentum 0.001 --bn-eps 0.001 --warmup-prefix
```

You may also refer to `evaluation/pytorch-image-models-main/gram_export/csco` and `evaluation/scripts` for other variants, such as the use of SE/Swish options and non AutoAugment training pipelines. For example, if you want to train the best model with SE, Hard-Swish and autoaugment, you may use the following script:

```sh
./distributed_train.sh 4 /imagenet/ --model gram224 -b 192 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 3e-5 --drop 0.0 --drop-path 0.0 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 --gram_proto gram_export/csco/csco_im_se_large.prototxt --data-dir ./data/ImageNet --bn-momentum 0.001 --bn-eps 0.001 --warmup-prefix
```

The results are shown as follows:

| Model Name           | MACS(G) | Params(M) | Top-1 Err. (%) | Top-5 Err. (%) |
|----------------------|---------|-----------|----------------|----------------|
| DARTS                | 574     | 4.7       | 26.7           | 8.7            |
| CSCO                 | 598     | 5.7       | 23.3           | 6.7            |
| CSCO + Swish/SE + AA | 602     | 6.8       | 21.9           | 6.0            |