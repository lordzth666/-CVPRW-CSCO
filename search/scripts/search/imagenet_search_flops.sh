C=$1

python -u mcmc_explore.py \
    --accuracy_pretrain_ckpt_path ./predictor/imagenet_acc_aug.pt \
    --yaml_cfg ./yml_config/imagenet/imagenet_eval_ft.yml  \
    --metagraph_path ./exps-0213/hybnas-imagenet-search/hybnas-imagenet.metagraph \
    --task imagenet \
    --latency_pretrain_ckpt_path ./predictor/imagenet_flops.pt \
    --constraint $C \
    --proto_dir ./imagenet_protos/imagenet_protos_flops_$C/ \
    --sensitivity 0.01 --batch_size 128 --alpha -0.5 --num_rounds 10000 --burn_in 10000