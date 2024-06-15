python -u imagenet_inference_sample.py --task imagenet \
    --model_dir ../exps-0213/hybnas-imagenet-search-flops \
    --yaml_cfg imagenet_search.yml \
    --budget 50000 \
    --task imagenet \
    --profile_only \
    --worker_id 0
