
C=$1
python mcmc_explore.py \
    --accuracy_pretrain_ckpt_path ./predictor/cifar10_acc_aug.pt  \
    --yaml_cfg ./cifar10_search.yml  \
    --metagraph_path ./exps-0213/hybnas-cifar10-search/hybnas-cifar10.metagraph \
    --task cifar10 \
    --proto_dir ./cifar10_protos/cifar10_protos_$C/ \
    --sensitivity 0.006 --batch_size 64 --alpha -1.0 --num_rounds 20000 \
    --latency_pretrain_ckpt_path ./predictor/cifar10_flops.pt \
    --constraint $C
