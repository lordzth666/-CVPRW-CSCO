from src.proto.lazy_loader import *

def make_divisible(n, divided_by=8):
    m =  divided_by * round(n / divided_by)
    return m

class ProtoWriter:
    def __init__(self, name=None):
        self.proto_list = []
        self.name = name
        self.in_name = None

    def add_header(self, task='cifar10'):
        if task in headers["native"].keys():
            proto, self.in_name = headers["native"][task](self.name)
            self.add(proto)

    def add(self, proto):
        self.proto_list.append(proto)

    def finalized(self, task='cifar10', out_name=None):
        if task in finals["native"].keys():
            proto = finals["native"][task](out_name)
            self.add(proto)
        else:
            raise NotImplementedError
        pass

    def scaling(self, factor, divisible_by=8):
        for id in range(len(self.proto_list)):
            for item_id in range(len(self.proto_list[id])):
                if len(self.proto_list[id][item_id])<7:
                    continue
                if self.proto_list[id][item_id][:7] == 'filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'filters=%d' %make_divisible(fnum*factor, divisible_by)
                if self.proto_list[id][item_id][:12] == 'base_filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'base_filters=%d' %make_divisible(fnum*factor, divisible_by)
                if self.proto_list[id][item_id][:11] == 'out_filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'out_filters=%d' %make_divisible(fnum*factor, divisible_by)

    def dump(self, fp):
        for out in self.proto_list:
            for item in out:
                fp.write(item + "\n")

        pass

    def _set_global_properties(self, property, value, exclude_property=[], count=-1):
        """
        Set global property of one specific value
        :param property:
        :param value:
        :param count: maximum counts for setting.
        :return: None
        """
        property_length = len(property)
        cnt = 0
        for id in range(len(self.proto_list)):
            if cnt == count:
                break
            for item_id in range(len(self.proto_list[id])):
                if len(self.proto_list[id][item_id])<property_length:
                    continue
                if self.proto_list[id][item_id][:property_length] == property:
                    rvalue = self.proto_list[id][item_id].split("=")[1]
                    if rvalue in exclude_property:
                        continue
                    self.proto_list[id][item_id] = '%s=%s' %(property, value)
                    cnt += 1
        pass

    def set_global_regularization(self, reg_value):
        """
        :param reg_value:
        :return:
        """
        self._set_global_properties("regularizer_strength", reg_value, exclude_property=["0"])

    def set_global_dropout(self, dropout_value):
        """
        :param dropout_value:
        :return:
        """
        self._set_global_properties('dropout', dropout_value)

    def set_bias_flag(self, flag):
        """
        :param flag:
        :return:
        """
        self._set_global_properties('use_bias', flag)

    def set_batchnorm_flag(self, flag):
        """
        Setting the batchnorm flag.
        :param flag: True/False
        :return:
        """
        self._set_global_properties("batchnorm", flag)

    def set_activation(self, activation):
        """
        Set the activation function of each layer in proto.
        :param activation: activation name.
        :return:
        """
        self._set_global_properties('activation', activation, exclude_property=['linear', 'softmax'])
        pass