import os
import torch.nn as nn

from src.utils.basic_layers import *
from src.utils.proto_mapper import ProtoMapper
from src.utils.block_manager import channel_block_manager, get_output_channels

class Architecture(nn.Module):
    def __init__(
            self, 
            prototxt, 
            in_channels=3, 
            ignore_warning=True,
            bn_momentum: float = 0.1,
            bn_epsilon: float = 1e-5,
            drop_connect_rate: float = 0.0,
            dropout_rate: float = 0.2,
            se_intra_activation_fn: str = "relu",
            se_output_activation_fn: str = "sigmoid",
        ):
        super().__init__()
        self.prototxt = prototxt
        mapped_protos = ProtoMapper(self.prototxt).parse_proto()
        self.in_channels = {'input': in_channels}
        self.acc_strides = {'input': 1}

        self.layers = nn.ModuleList([])
        self.layer_outputs = []
        self.layer_operands = []

        self.model_name = None
        self.save_path = None
        self.pretrain_path = None
        
        # Add configurations.
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.se_intra_activation_fn = se_intra_activation_fn
        self.se_output_activation_fn = se_output_activation_fn
        
        print("Building architecture with the following configs...")
        print(vars(self))

        self.ignore_warning = ignore_warning

        for block_proto in mapped_protos:
            if block_proto['type'] == "Model":
                self.model_name = get_kwarg_with_default(block_proto, 'name', None)
                self.save_path = os.path.join("models", self.model_name)
                self.pretrain_path = get_kwarg_with_default(block_proto, "pretrain", None)
                continue

            if not block_proto['type'] in layer_mapper.keys():
                if not ignore_warning:
                    print("%s not supported!" %block_proto['type'])
                continue

            input_name = block_proto['input']
            output_name = block_proto['name']

            op_type = block_proto['type']
            if op_type == "Dense":
                output_channels = get_kwarg_with_default(block_proto, 'units', 7)
            else:
                output_channels = get_kwarg_with_default(block_proto, 'filters', 7)
                
            # Override Dropout on ImageNet.
            if op_type == "Dropout":
                block_proto["dropout"] = self.dropout_rate

            strides = get_kwarg_with_default(block_proto, 'strides', 1)

            if isinstance(input_name, str):
                if input_name in self.in_channels.keys():
                    in_channels = get_kwarg_with_default(self.in_channels, input_name, -1)
                    cur_strides = get_kwarg_with_default(self.acc_strides, input_name, 1)
                else:
                    print("Warning. Tensor name %s is lingering and optimized in the whole flow." %input_name)
                    continue

                self.layers.append(
                    channel_block_manager(
                        op_type, block_proto, in_channels,
                        bn_momentum=self.bn_momentum,
                        bn_epsilon=self.bn_epsilon,
                        drop_connect_rate=self.drop_connect_rate,
                        dropout_rate=self.dropout_rate,
                        se_intra_activation_fn=self.se_intra_activation_fn,
                        se_output_activation_fn=self.se_output_activation_fn,
                    )
                )
                self.layer_outputs.append(output_name)
                self.layer_operands.append(input_name)

                depth_multiplier = get_kwarg_with_default(block_proto, 'depth_multiplier', 1)
                depthwise_multiplier = get_kwarg_with_default(block_proto, 'depthwise_multiplier', 1)

                # Finally, add the output layers.
                self.in_channels[output_name] = get_output_channels(op_type, input_name, self.in_channels, output_channels,
                                                                    depth_multiplier=depth_multiplier,
                                                                    depthwise_multiplier=depthwise_multiplier)
                self.acc_strides[output_name] = cur_strides * strides

            elif isinstance(input_name, list):
                op_type = block_proto['type']
                if op_type == "Add":
                    if len(input_name) != 2:
                        raise NotImplementedError("Input list should contain exactly 2 tensors.")
                    if input_name[0] in self.in_channels.keys():
                        op1_channels = get_kwarg_with_default(self.in_channels, input_name[0], -1)
                        cur_strides_1 = get_kwarg_with_default(self.acc_strides, input_name[0], 1)
                    else:
                        print("Warning. Tensor name %s is lingering and optimized in the whole flow." % input_name[0])
                        continue
                    if input_name[1] in self.in_channels.keys():
                        op2_channels = get_kwarg_with_default(self.in_channels, input_name[1], -1)
                        cur_strides_2 = get_kwarg_with_default(self.acc_strides, input_name[1], 1)
                    else:
                        print("Warning. Tensor name %s is lingering and optimized in the whole flow." % input_name[1])
                        continue
                    self.layers.append(channel_block_manager(
                            op_type, block_proto, op1_channels, op2_channels,
                            inferred_strides=cur_strides_2 // cur_strides_1,
                            bn_momentum=self.bn_momentum,
                            bn_epsilon=self.bn_epsilon,
                            drop_connect_rate=self.drop_connect_rate,
                            dropout_rate=self.dropout_rate,
                            se_intra_activation_fn=self.se_intra_activation_fn,
                            se_output_activation_fn=self.se_output_activation_fn,
                        )
                    )
                    self.layer_outputs.append(output_name)
                    self.layer_operands.append(input_name)
                    # Finally, add the output layers.
                    self.in_channels[output_name] = get_output_channels(op_type, input_name,
                                                                        self.in_channels, output_channels)
                    self.acc_strides[output_name] = cur_strides_2
                elif op_type == 'Concat':
                    if input_name[0] in self.in_channels.keys():
                        in_channels = get_kwarg_with_default(self.in_channels, input_name[0], -1)
                        cur_strides = get_kwarg_with_default(self.acc_strides, input_name[0], 1)
                    else:
                        print("Warning. Tensor name %s is lingering and optimized in the whole flow." % input_name)
                        continue

                    self.layers.append(channel_block_manager(op_type, block_proto, in_channels))
                    self.layer_outputs.append(output_name)
                    self.layer_operands.append(input_name)
                    # Finally, add the output layers.
                    self.in_channels[output_name] = get_output_channels(
                        op_type, input_name, self.in_channels,
                        output_channels,
                    )
                    self.acc_strides[output_name] = cur_strides
                else:
                    raise NotImplementedError("Op %s does not belong to any category!" %op_type)


    def _get_operands(self, canvas, operand_name):
        """
        Get the operands in the canvas.
        :param operand_name: string or list.
        :return: operand or operand list.
        """
        if isinstance(operand_name, str):
            return canvas[operand_name]
        else:
            operands = []
            try:
                for name in operand_name:
                    operands.append(canvas[name])
                return operands
            except Exception:
                raise NotImplementedError

    def forward(self, x):
        # Keeping track of the Res block for drop-path.
        canvas = {'input': x}
        assert len(self.layers) != 0, NotImplementedError("The number of layers must not be 0.")
        for i in range(len(self.layers)):
            layer = self.layers[i]
            operands = self._get_operands(canvas, self.layer_operands[i])
            # print("Processing Layer %s ..."%(layer))
            # print(operands.size())
            # print(layer)
            out = layer(operands)
            canvas[self.layer_outputs[i]] = out
        # Final output is returned
        return canvas[self.layer_outputs[-1]]

    def forward_with_names(self, x, names):
        # Keeping track of the Res block for drop-path.
        tensors = []
        canvas = {'input': x}
        assert len(self.layers) != 0, NotImplementedError("The number of layers must not be 0.")
        for i in range(len(self.layers)):
            layer = self.layers[i]
            operands = self._get_operands(canvas, self.layer_operands[i])
            # print("Processing Layer %s ..."%(layer))
            # print(operands.size())
            # print(layer)
            out = layer(operands)
            canvas[self.layer_outputs[i]] = out
            if self.layer_outputs[i] in names:
                tensors.append(canvas[self.layer_outputs[i]])
            if self.layer_outputs[i] == names[-1]:
                break
        # Final output is returned
        return tuple(tensors)

    def set_drop_connect_rate(self, drop_connect_rate: float = 0.0):
        print(f"Setting drop path to {drop_connect_rate}!")
        for layer in self.layers:
            if isinstance(layer, Add):
                layer.set_drop_connect_rate(drop_connect_rate)
