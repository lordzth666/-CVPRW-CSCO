python -u cifar10_utils/train_cifar10_with_tower_logits.py \
    --yaml_cfg ./yml_config/cifar10/cifar10_eval_darts_600e_1GPU_noAMP_noEMA.yml \
    --prototxt ./acenet_builder/best_configs/cifar10/bybnas_best_cifar10_mcmc_aug_param_wm400.prototxt