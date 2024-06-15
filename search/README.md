# Search Code

This folder contains the search code needed to perform CSCO search. To enable search, you should first `cd search`.

## Established database
Please refer to `csco_database` for an established database containing architecture-performance pairs and the related statistics. You may simply look at the database via search/notebooks/view_database.ipynb.

## Sample from scratch.
To sample new architectures from scratch, you may refer to the following script in `search/scripts/sample`:

```
python main_sample.py --task cifar10 \
    --model_dir ../exps-0213/hybnas-cifar10-search \
    --yaml_cfg cifar10_search.yml \
    --budget 400 \
    --worker_id $1
```

As we will utilize accuracy/latency/FLOPs predictor to achieve co-design and joint search, please refer to ``search/scripts/sample/slurm_imagenet_sample_flops.sh`` to see commands.

## Graph Isomorphism

You can use the script in `search/core/predictor/graph_isomorphism.py` to perform graph isomorphism. Specifically, you may find this code snipet:
```python
def main(args):
    num_nodes = 13
    task = "cifar10"
    exp_dir = "./exps-0213/hybnas-{}-search/".format(task)
    exp_aug_dir = "./exps-0213/hybnas-{}-search-augmented/".format(task)
    os.makedirs(exp_aug_dir, exist_ok=True)
```

Simply replace `exp_dir` to be the records before graph isomorphism, and `exp_aug_dir` to match the output directory of graph isomorphism.

## Predictor training.
You may train a neural predictor to map records towards performance, and make predictions for future explorations. For example,
```sh
CUDA_VISIBLE_DEVICES=0 python predictor/build_predictor.py --record_dir ./exps-0213/hybnas-cifar10-search/ --task cifar10 --save_ckpt_path ./predictor/cifar10_acc.pt --learning_rate 0.04 --batch_size 128 --weight_decay 3e-4 --ranking_loss_margin 0.0 --num_epochs 300  --all_in
```

Note that `--all_in` specifies all examples for predictor training. When you are experimenting with new GI features/predictor settings, disable `all_in` so you can get an accurate evaluation of Kendall's tau.

## MH-ES
You may refer to the scripts in `scripts/search` to see MHES options. For example,
```
C=$1
python mhes_explore.py \
    --accuracy_pretrain_ckpt_path ./predictor/cifar10_acc_aug.pt  \
    --yaml_cfg ./cifar10_search.yml  \
    --metagraph_path ../exps-0213/hybnas-cifar10-search/hybnas-cifar10.metagraph \
    --task cifar10 \
    --proto_dir ./cifar10_protos/params_mcmc/cifar10_protos_$C/ \
    --sensitivity 0.01 --batch_size 64 --alpha -1.0 --num_rounds 15000 \
    --latency_pretrain_ckpt_path predictor/cifar10_params.pt \
    --constraint $C
```
Here, $1 is the normalized constrained (i.e., 0-1) you should choose to enforce latency/FLOPs contrainst. Typically 0.3~0.4 should be a good number.

## Building the final models.
Check `core/net_builder`. DO NOT forget to replace the configs to reflect the best protos in discovery.