# Contrib layers in the format of caffe prototxt
from gram_export.gram.proto.caffe_layer import *

#TODO: Build architectures with basic layer definition in engine/proto/caffe_layer.py
#TODO: add them to the library in the lazy_loader.
#TODO: to call them, use G.set_proto_backend("caffe") before instantializing the proto class.
#TODO: use G.set_proto_backend("native") to swtich back to original form of prototxt.

def Caffe_Cifar10_header(model_name='cifar10-model'):
    """
    Build pre-swiftnet layers of cifar-10 model in caffe prototxt using predefined proto functions in engine/proto/caffe_layer.py
    :param model_name: Name of the model.
    :return: ret: returned prototxt; <str>: final layer name of the pre-swiftnet arch which will be fed into swiftnet.
    """
    ret = []
    ret.append('name: "CIFAR10_quick"')
    ret+=Caffe_CIFAR10_Input_proto(phase='train')
    ret+=Caffe_CIFAR10_Input_proto(phase='test')

    #TODO: build the tower. Use ret += (***_PROTO) to stack protos. e.g. ret += Caffe_conv_proto(**kwargs)
    return ret, "input"


def Caffe_Cifar10_final(out_name=None):
    """
    Build post-swiftnet layers of cifar-10 model in caffe prototxt using predefined proto functions in engine/proto/caffe_layer.py
    :param out_name: Output name of swiftnet which will be fed into post-swiftnet archs.
    :return: ret: returned prototxt
    """
    ret = []
    #ret += Flatten_proto(name='flatten',
    #                        input=out_name)
    ret += Caffe_InnerProduct_proto(name='ip1',
                       input='out_name',
                       num_output=64)
    ret += Caffe_InnerProduct_proto(name='ip2',
                       input='ip1',
                       num_output=10)
    ret += Caffe_Accuracy_proto(name='accuracy',
                          input='ip2')
    ret += Caffe_SoftmaxWithLoss_proto(name='loss',
                         ip2='ip2')

    return ret

    #TODO: build the tower. Use ret += (***_PROTO) to stack protos. e.g. ret += Caffe_conv_proto(**kwargs)
