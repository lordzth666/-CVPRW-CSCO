from src.utils.architecture import Architecture

def build_model_with_yaml_config(config):
    return Architecture(
        prototxt=config.prototxt,
        bn_momentum=float(config.bn_momentum),
        bn_epsilon=float(config.bn_epsilon),
        drop_connect_rate=float(config.drop_connect),
        dropout_rate=0.0, # Deprecated
        se_intra_activation_fn=config.se_intra_activation_fn,
        se_output_activation_fn=config.se_output_activation_fn,
    )