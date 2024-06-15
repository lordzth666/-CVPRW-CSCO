import yaml

class Dict2Attr(object):
    def __init__(self, d):
        self.__dict__ = d

def preprocess_config(config):
    """
    Preprocessing yaml config if needed.
    :param config: Raw config.
    :return: processed config.
    """
    return config

def apply_config_to_backend(config):
    config.lr = float(config.lr)
    config.weight_decay = float(config.weight_decay)
    config.num_classes = int(config.num_classes)
    try:
        config.downsample_ratio = float(config.downsample_ratio)
    except Exception:
        config.downsample_ratio = 1.0
    try:
        config.regularize_depthwise = int(config.regularize_depthwise)
    except Exception:
        config.regularize_depthwise = 1

    # Exponential moving average
    config.EMA = float(config.EMA)


def apply_config(config):
    apply_config_to_backend(config)

def load_and_apply_yaml_config(yaml_file):
    with open(yaml_file, 'r') as fp:
        Config = yaml.load(fp, yaml.Loader)
    config = Dict2Attr(Config)
    print(vars(config))
    processed_config = preprocess_config(config)
    apply_config(config)
    return processed_config
