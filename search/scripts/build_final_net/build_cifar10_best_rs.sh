python acenet_builder/build_best_cifar10.py \
    --proto_dir ./acenet_builder/best_configs/ \
    --yaml_cfg ./yml_config/cifar10/cifar10_eval.yml \
    --metagraph_path ./exps-0213/hybnas-cifar10-search/hybnas-cifar10.metagraph \
    --model_json_config acenet_builder/best_base_json/cifar10_best_rs.json \
    --depth 6 --width_multiplier 3.25 --dropout 0