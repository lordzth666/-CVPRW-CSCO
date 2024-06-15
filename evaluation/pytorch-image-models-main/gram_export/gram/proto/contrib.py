from gram_export.gram.proto.layer import *
from gram_export.gram.backend.backend import G

# RGB format
_imagenet_mean = [123.68, 116.78, 103.94]
_imagenet_std = [58.39, 57.12, 57.31]


def Cifar10_header(model_name='cifar10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[32, 32, 3], name='input')
    ret += Input_proto(input_shape=[G.num_classes], name='label')
    return ret, "input"


def EDA_incomplete_net_header(model_name='cifar10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[224, 224, 16], name='input')
    ret += Input_proto(input_shape=[1], name='label')
    return ret, "input"


def EDA_density_header(model_name='cifar10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[224, 224, 16], name='input')
    ret += Input_proto(input_shape=[56, 56, 1], name='label')
    return ret, "input"


def Cifar100_header(model_name='cifar100-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    # ret.append('pretrain=None')
    ret.append('')
    ret += Input_proto(input_shape=[32, 32, 3], name='input')
    ret += Input_proto(input_shape=[100], name='label')
    ret += Convolutional_Proto(name='conv_pre1',
                               kernel_size=3,
                               strides=1,
                               padding='SAME',
                               filters=32,
                               input='input',
                               activation='linear',
                               batchnorm=True,
                               regularizer_strength=1e-4,
                               use_bias=False)
    return ret, "conv_pre1"


def Cifar10_final(out_name=None):
    ret = []

    ret += Convolutional_Proto(name='logits_4d',
                               input=out_name,
                               filters=G.num_classes,
                               kernel_size=1,
                               activation='linear',
                               batchnorm=False,
                               use_bias=True,
                               use_dense_initializer=True)
    ret += Flatten_proto(name='logits',
                         input='logits_4d')
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def EDA_incomplete_net_final(out_name=None):
    ret = []
    ret += Convolutional_Proto(name='logits_4d',
                               input=out_name,
                               filters=1,
                               kernel_size=1,
                               activation='linear',
                               batchnorm=False,
                               use_bias=True,
                               use_dense_initializer=True)
    ret += Flatten_proto(name='logits',
                         input='logits_4d')
    ret += Output_proto(name="pred_labels",
                        input="logits")
    ret += SmoothedL1_proto(name='smoothedl1_loss',
                            input='logits',
                            labels='label',
                            delta=.01)
    ret += MSE_Proto(name='mse',
                     logits='logits',
                     labels='label')
    ret += MAE_Proto(name='mae',
                     logits='logits',
                     labels='label')
    return ret


def EDA_density_final(out_name=None):
    ret = []
    ret += Convolutional_Proto(name='logits',
                               input=out_name,
                               filters=1,
                               kernel_size=1,
                               activation='linear',
                               batchnorm=False,
                               use_bias=True,
                               use_dense_initializer=True)
    ret += Output_proto(name="pred_labels",
                        input="logits")
    ret += BCELoss_proto(name='bce_loss',
                         input='logits',
                         labels='label')
    ret += BinaryAccuracy_proto(name='mae',
                                logits='logits',
                                labels='label')
    return ret


def Cifar100_final(out_name=None):
    ret = []
    ret += Dense_proto(name=out_name,
                       input='flatten',
                       units=100,
                       activation='linear',
                       batchnorm=False,
                       use_bias=True,
                       regularizer_scale=G.DEFAULT_REG)

    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def SVHN_header(model_name='cifar10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[32, 32, 3], name='input')
    ret += Input_proto(input_shape=[10], name='label')
    return ret, "input"


def SVHN_final(out_name=None):
    ret = []
    ret += Dense_proto(name='logits',
                       input='flatten',
                       units=10,
                       activation='linear',
                       batchnorm=True)
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')
    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def Fashion_MNIST_header(model_name='fmnist-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[28, 28, 1], name='input')
    ret += Input_proto(input_shape=[10], name='label')
    return ret, "input"


def Fashion_MNIST_final(out_name=None):
    ret = []
    ret += Dense_proto(name='logits',
                       input='flatten',
                       units=G.num_classes,
                       activation='linear',
                       batchnorm=True)
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')
    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def MNIST_header(model_name='mnist-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[28, 28, 1], name='input')
    ret += Input_proto(input_shape=[G.num_classes], name='label')
    return ret, "input"


def MNIST_final(out_name=None):
    ret = []
    ret += Dense_proto(name='logits',
                       input=out_name,
                       units=10,
                       activation='linear',
                       batchnorm=True)
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')
    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def STL10_header(model_name='stl-10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[96, 96, 3], name='input')
    ret += Input_proto(input_shape=[10], name='label')

    return ret, "input"


def STL10_final(out_name=None):
    ret = []
    ret += Flatten_proto(name='flatten',
                         input=out_name)

    ret += Dense_proto(name='logits',
                       input='flatten',
                       units=10,
                       activation='linear',
                       batchnorm=True)
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')
    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


# For latest work (e.g. ACE, the stem conv size is fixed to 16.)
# For mobilenetv3-large, the stem conv size is fixed to 16.
# For other mobile networks, the stem conv size is fixed to 32.
def ImageNet_header(model_name='imagenet-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    ret.append('')
    ret += Input_proto(input_shape=[224, 224, 3],
                       name='input')
    ret += Input_proto(input_shape=[G.num_classes], name='label')
    out = "input"
    return ret, out


def ImageNet_header_demo(model_name='imagenet-model',
                         num_classes=1001,
                         size=224):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' % model_name)
    ret.append('')

    ret += Input_proto(input_shape=[size, size, 3],
                       name='input')
    ret += Input_proto(input_shape=[num_classes], name='label')

    """
    ret += Convolutional_Proto(name='conv_pre1',
                               kernel_size=3,
                               strides=2,
                               padding='SAME',
                               filters=16,
                               input='input',
                               activation='linear',
                               batchnorm=False,
                               regularizer_strength=1e-12,
                               use_bias=False)"""
    return ret, "input"


def ImageNet_final(out_name=None):
    ret = []
    ret += Convolutional_Proto(name='logits_4d',
                               input=out_name,
                               filters=G.num_classes,
                               kernel_size=1,
                               activation='linear',
                               batchnorm=False,
                               use_bias=True,
                               use_dense_initializer=True)
    ret += Flatten_proto(name='logits',
                         input='logits_4d')
    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label',
                         label_smoothing=0.1)

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')

    ret += TopkAcc_proto(name='top_5_acc',
                         logits='logits',
                         labels='label',
                         k=5)
    return ret


# Please note that model header/final will contain general information from now on.
def ImageNet_final_demo(out_name=None, size=224,
                        num_classes=1001):
    ret = []
    ret += Convolutional_Proto(name='conv1x1',
                               input=out_name,
                               filters=1280,
                               regularizer_strength=1e-6,
                               activation='relu',
                               kernel_size=1,
                               batchnorm=True,
                               use_bias=False)
    ret += GlobalAvgPool_proto(name='avg_1k',
                               input='conv1x1')
    ret += Flatten_proto(name='flatten',
                         input='avg_1k')
    ret += Dense_proto(name='logits',
                       input='flatten',
                       units=num_classes,
                       regularizer_scale=1e-6,
                       activation='softmax',
                       batchnorm=False)
    ret += Output_proto(name='output',
                        input='logits')

    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')

    ret += TopkAcc_proto(name='top_5_acc',
                         logits='logits',
                         labels='label',
                         k=5)
    return ret
