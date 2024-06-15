def bool_str(var):
	if var == "True":
		return True
	else:
		return False

def integer_list(var):
	var = var.strip("[").strip("]")
	var_list = var.split(",")
	return [int(var_) for var_ in var_list]

def float_list(var):
	var_ = var.replace("[", '')
	var_ = var_.replace("]", '')
	var_ = var_.replace("'", '')
	var_ = var_.replace(" ", '')
	var_list = var_.split(",")
	return [float(var_) for var_ in var_list]

def str_or_list(var):
	var_ = var.replace("[", '')
	var_ = var_.replace("]", '')
	var_ = var_.replace("'", '')
	var_ = var_.replace(" ", '')
	if var_.find(",") != -1:
		return(var_.split(","))
	else:
		return var_

class ProtoType:
	def __init__(self, processing_lists):
		self.keyword_type_maps = {
			'name': str,
			'input': str_or_list,
			'pretrain': str,
			'input_shape': integer_list,
			'filters': int,
			'units': int,
			'labels': str,
			'kernel_size': int,
			'strides': int,
			'padding': str,
			'dtype': str,
			'mean': float,
			'std': float,
			'activation': str,
			'regularizer_strength': float,
			'dropout': float,
			'use_bias': bool_str,
			'trainable': bool_str,
			'batchnorm': bool_str,
			'pool_size': int,
			'label_smoothing': float,
			'logits': str,
			'use_dense_initializer': bool_str,
            'depth_multiplier': int,
            "depthwise_multiplier": int,
            'se_ratio': float,
            'k': int,
		}
		self.processing_lists = processing_lists

	def __call__(self):
		return_dict = {}
		# 0. Type of the operation.
		return_dict['type'] = self.processing_lists[0][1:-1]
		for id in range(1, len(self.processing_lists)):
			key = self.processing_lists[id].split("=")[0]
			typedef = self.keyword_type_maps[key]
			return_dict[key] = typedef(self.processing_lists[id].split("=")[-1])
		return return_dict
