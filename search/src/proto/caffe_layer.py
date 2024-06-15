# Basic layer protos in the form of caffe
# Refer to layer.py for details.

## implemented detail:
## xavier initialization by default.
## combine conv, activation, bn into one function

## TODO
## add lr_mult

from src.backend.backend import G

def Caffe_conv_proto(name,
                    input,
                    filters,
                    kernel_size,
                    strides,
                    padding,
                    activation='relu',
                    batchnorm=False,
                    dropout=0.00
              #regularizer_strength=G.CONV_DEFAULT_REG,

              ):
    """
    :param name: name of the conv op
    :param input: input (tensor) name
    :param filters: filter size
    :param kernel_size: kernel size
    :param strides: conv stride
    :param padding: conv padding
    :param activation: conv activation function.
    :param batchnorm: batchnorm option
    :param regularizer_strength: regularization strength
    :return:
    """

    ret = []
    # conv
    ret.append('layer {')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    ret.append('  name: "%s"' %name)
    ret.append('  type: "Convolution"')
    ret.append('  convolution_param {')
    ret.append(    'num_output: %d' %filters)
    ret.append(    'pad: {}'.format(padding) )
    ret.append(    ' kernel_size: %d' %kernel_size)
    ret.append(    ' stride: %d' %strides)
    ret.append(    ' weight_filler: {')
    ret.append(    '   type: "xavier"')
    ret.append(    ' }')
    ret.append(    ' bias_filler: {')
    ret.append(    '   type: "constant"')
    ret.append(    '   value: 0')
    ret.append(    ' }')
    ret.append('  }')
    ret.append('}')
    ret.append('')

    # activation
    ret.append('layer {')
    ret.append('  name: "%s/act"' % name)
    ret.append('  type: "%s"' % activation)
    ret.append('  bottom: "%s"' % name)
    ret.append('  top: "%s"' % name)
    ret.append('}')
    ret.append('')

    # batchnorm
    if batchnorm:
        ret.append('layer {')
        ret.append('  name: "%s/bn"' % name)
        ret.append('  type: "BatchNorm"')
        ret.append('  bottom: "%s"' % name)
        ret.append('  top: "%s"' % name)
        # start of batchnorm params
        ret.append('  batch_norm_param {')
        ret.append('    use_global_status: true')
        ret.append('  }')
        ret.append('}')
        ret.append('')

    return ret
#
def Caffe_Identity_proto(name,
                         input):
    ret = []
    ret.append('layer {')
    ret.append('  name: "%s/act"' % name)
    ret.append('  type: "Scale"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    ret.append('  scale_param {')
    ret.append('  filler {')
    ret.append('  type: "constant"')
    ret.append('    value: 1')
    ret.append('  }')
    ret.append('}')
    ret.append('')
    return ret

def Caffe_Sepconv_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        regularizer_strength=G.SEPCONV_DEFAULT_REG,
                        dropout=0.00):
    ## TODO not done yet
    raise NotImplementedError
    ret = []
    # conv
    ret.append('layer {')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    ret.append('  name: "%s/dw"' %name)
    ret.append('  type: "Convolution"')
    ret.append('  convolution_param {')
    ret.append(    'num_output: %d' %filters)
    ret.append(    'pad: {}'.format(padding) )
    ret.append(    ' kernel_size: %d' %kernel_size)
    ret.append(    ' stride: %d' %strides)
    ret.append(    ' weight_filler: {')
    ret.append(    '   type: "xavier"')
    ret.append(    ' }')
    ret.append(    ' bias_filler: {')
    ret.append(    '   type: "constant"')
    ret.append(    '   value: 0')
    ret.append(    ' }')
    ret.append('  }')
    ret.append('}')
    ret.append('')

    # activation
    ret.append('layer {')
    ret.append('  name: "%s/act"' % name)
    ret.append('  type: "%s"' % activation)
    ret.append('  bottom: "%s"' % name)
    ret.append('  top: "%s"' % name)
    ret.append('}')
    ret.append('')

    # batchnorm
    if batchnorm:
        ret.append('layer {')
        ret.append('  name: "%s/bn"' % name)
        ret.append('  type: "BatchNorm"')
        ret.append('  bottom: "%s"' % name)
        ret.append('  top: "%s"' % name)
        # start of batchnorm params
        ret.append('  batch_norm_param {')
        ret.append('    use_global_status: true')
        ret.append('  }')
        ret.append('}')
        ret.append('')

    return ret
"""
    TODO: Depthwise multiplier is 1 by default.
    :param name: name of the sepconv op
    :param input: input (tensor) name
    :param filters: filter size
    :param kernel_size: kernel size
    :param strides: conv stride
    :param padding: conv padding
    :param activation: conv activation function.
    :param batchnorm: batchnorm option
    :param regularizer_strength: regularization strength
    :return:

"""
#TODO: Implement all required layers in engine/proto/layer.py with similar format.


def Caffe_Concat_proto(name,
                       input):
    ret = []
    ret.append('layer {')
    ret.append('  name: "%s"' % name)
    ## put them all in list first. Will parse later
    ret.append('  bottom: %' % input)
    ret.append('  top: "%s"' % name)
    ret.append('  type: "Concat"')
    ret.append('  concat_param {')
    ret.append('    axis: 1')
    ret.append('  }')
    ret.append('')

    return ret
def Caffe_MaxPool_proto(name,
                        input,
                        pool_size=2,
                        strides=2,
                        padding='SAME'):
    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "%s"' % name)
    ret.append('  type: "Pooling"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    # start of params
    ret.append('  pooling_param {')
    ret.append('    pool: MAX')
    ret.append('    kernel_size: %d' %pool_size)
    ret.append('    stride : %d' %strides)
    # end of paprams
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_AvePool_proto(name,
              input,
              kernel_size,
              strides):
    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "%s"' % name)
    ret.append('  type: "Pooling"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    # start of params
    ret.append('  pooling_param {')
    ret.append('    pool: AVE')
    ret.append('    kernel_size: %d' %kernel_size)
    ret.append('    stride : %d' %strides)
    # end of paprams
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_CIFAR10_Input_proto(phase):

    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "cifar"')
    ret.append('  type: "Data"')
    ret.append('  bottom: "data"')
    ret.append('  top: "label"')
    # start of params
    ret.append('  include {')
    if phase == 'train':
        ret.append('    phase: TRAIN')
    elif phase =='test':
        ret.append('    phase: TEST')
    # end of paprams
    ret.append('  }')
    # start of transform_param
    ret.append('  transform_param {')
    ret.append('    mean_file: "examples/cifar10/mean.binaryproto')
    # end of transform_param
    ret.append('  }')
    # start of data_param
    ret.append('  data_param {')
    if phase == 'train':
        ret.append('    source: "examples/cifar10/cifar10_train_lmdb"')
    elif phase == 'test':
        ret.append('    source: "examples/cifar10/cifar10_test_lmdb"')
    ret.append('    batch_size: 100')
    ret.append('    backend: LMDB')
    # end of transform_param
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_ssd_Input_proto(name,
              input,
              filters,
              kernel_size,
              phase='TRAIN',
              strides=1,
              padding='SAME',
              activation='relu',
              batchnorm=False,
              regularizer_strength=G.CONV_DEFAULT_REG,
              hyper=False,
              hyper_zdims=None,
              hyper_basic_block_size=None,
              hyper_hidden=None):
    """
    TODO: finish this with https://github.com/newwhitecheng/caffe/blob/ssd/models/VGGNet/VOC0712/SSD_300x300/train.prototxt
    """

    """
        Adding things to ret in caffe format. e.g. ret.append('"name: "mnist"
          type: "Data"\ntransform_param {\nscale: 0.00390625\n}')
    """

    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "data"')
    ret.append('  type: "AnnotatedData"')
    ret.append('  bottom: "data"')
    ret.append('  top: "label"')
    # start of params
    ret.append('  inlcude {')
    ret.append('    phase: "%s"' % phase)
    # end of paprams
    ret.append('  }')
    # start of transform params
    ret.append('  transform_param {')
    ret.append('    mirror: true')
    ret.append('    mean_value: 104.0')
    ret.append('    mean_value: 117.0')
    ret.append('    mean_value: 123.0')
    # end of transform paprams
    ret.append('    resize_param {')
    ret.append('      prob: 1.0')
    ret.append('      resize_mode: WARP')
    ret.append('      height: 300')
    ret.append('      width: 300')
    ret.append('      interp_mode: LINEAR')
    ret.append('      interp_mode: AREA')
    ret.append('      interp_mode: NEAREST')
    ret.append('      interp_mode: CUBIC')
    ret.append('      interp_mode: LANCZOS4')
    ret.append('    }')
    ret.append('      emit_constraint {')
    ret.append('        emit_type: CENTER')
    ret.append('      }')

    ret.append('  }')
    ret.append('  data_param {')
    ret.append('    source: "examples/VOC0712/VOC0712_trainval_lmdb"')
    ret.append('    batch_size: 16')
    ret.append('    backend: LMDB')
    ret.append('  }')
    ret.append('  label_map_file: "data/VOC0712/labelmap_voc.prototxt"')


    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_Flatten_proto(name,
              input):
    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "%s"' % name)
    ret.append('  type: "Flatten"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    # start of params
    ret.append('  flatten_param {')
    ret.append('    axis: 1')
    # end of paprams
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_InnerProduct_proto(name,
              input,
              num_output):
    # TODO: fill in some params like bias filller, lr_mult
    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "%s"' % name)
    ret.append('  type: "InnerProduct"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  top: "%s"' % name)
    # start of params
    ret.append('  inner_product_param {')
    ret.append('    num_output: %d' % num_output)
    # end of params
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_Accuracy_proto(name,
              input):
    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "accuracy"' % name)
    ret.append('  type: "Accuracy"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  bottom: "label"')
    ret.append('  top: "accuracy"')
    # start of phase
    ret.append('  include {')
    ret.append('    phase: TEST')
    # end of paprams
    ret.append('  }')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret

def Caffe_SoftmaxWithLoss_proto(name,
              input):

    ret = []
    # start of layer
    ret.append('layer {')
    ret.append('  name: "loss"' % name)
    ret.append('  type: "SoftmaxWithLoss"')
    ret.append('  bottom: "%s"' % input)
    ret.append('  bottom: "label"')
    ret.append('  top: "loss"')
    # end of layer
    ret.append('}')
    ret.append('')

    return ret