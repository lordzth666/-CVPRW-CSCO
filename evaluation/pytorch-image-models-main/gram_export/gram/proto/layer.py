from gram_export.gram.backend.backend import G

def Activation_Proto(name,
                     input,
                     activation='linear'):
    ret = []
    ret.append('[Activation]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('')

    return ret


def BatchNorm_Proto(name,
                    input,
                    activation='linear'):
    ret = []
    ret.append('[BatchNorm]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('')
    return ret

def Convolutional_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        regularizer_strength=1e-5,
                        dropout=0.00,
                        use_bias=True,
                        trainable=True,
                        use_dense_initializer=False):
    ret = []
    ret.append('[Convolutional]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)
    ret.append('use_dense_initializer=%s' %use_dense_initializer)
    ret.append('batchnorm=%s' %batchnorm)
    ret.append('')
    return ret


def TransposedConv_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        regularizer_strength=1e-5,
                        dropout=0.00,
                        use_bias=True,
                        trainable=True,
                        use_dense_initializer=False):
    ret = []
    ret.append('[TransposedConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)
    ret.append('use_dense_initializer=%s' %use_dense_initializer)
    ret.append('batchnorm=%s' %batchnorm)
    ret.append('')
    return ret

def ElasticConvolutional_Proto(name,
                        input,
                        shared_kernel_shape,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        regularizer_strength=1e-5,
                        dropout=0.00,
                        use_bias=True,
                        trainable=True,
                        use_dense_initializer=False):
    ret = []
    ret.append('[ElasticConvolutional]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('shared_kernel_shape=%s' %shared_kernel_shape)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)
    ret.append('use_dense_initializer=%s' %use_dense_initializer)
    ret.append('batchnorm=%s' %batchnorm)
    ret.append('')
    return ret


def Convolutional1D_Proto(name,
                            input,
                            filters,
                            kernel_size,
                            strides=1,
                            padding='SAME',
                            activation='relu',
                            batchnorm=False,
                            regularizer_strength=1e-5,
                            dropout=0.00,
                            hyper=False,
                            hyper_zdims=None,
                            hyper_basic_block_size=None,
                            hyper_hidden=None,
                            use_bias=True,
                            trainable=True):
    ret = []
    ret.append('[Convolutional1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)

    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')
    if hyper:
        ret.append('hyper=True')
        ret.append('hyper_zdims=%d' % hyper_zdims)
        ret.append('hyper_basic_block_size=%s' % hyper_basic_block_size)
        ret.append('hyper_hidden=%d' % hyper_hidden)

    ret.append('')
    return ret


def SeparableConv_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        dropout=0.00,
                        regularizer_strength=1e-5,
                        depth_multiplier=1,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[SeparableConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('depth_multiplier=%d' %depth_multiplier)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%E' % regularizer_strength)
    ret.append('trainable=%s' %trainable)
    ret.append('dropout=%.3f' %dropout)
    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')

    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def DepthwiseConv_Proto(name,
                        input,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        depthwise_multiplier=1,
                        batchnorm=False,
                        dropout=0.00,
                        regularizer_strength=1e-5,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[DepthwiseConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('depthwise_multiplier=%d' %depthwise_multiplier)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%E' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('trainable=%s' %trainable)
    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')

    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret

def ElasticDepthwiseConv_Proto(name,
                        input,
                        shared_kernel_shape,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        depthwise_multiplier=1,
                        batchnorm=False,
                        dropout=0.00,
                        regularizer_strength=1e-5,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[ElasticDepthwiseConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('shared_kernel_shape=%s' %shared_kernel_shape)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('depthwise_multiplier=%d' %depthwise_multiplier)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%E' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('trainable=%s' %trainable)
    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')

    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def MaxPool_proto(name,
                  input,
                  pool_size=2,
                  strides=2,
                  padding='SAME'):
    ret = []
    ret.append('[MaxPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' %padding)
    ret.append('')
    return ret


def MaxPool1D_proto(name,
                  input,
                  pool_size=2,
                  strides=2,
                    padding='SAME'):
    ret = []
    ret.append('[MaxPool1D]')
    ret.append('name=%s' %name)
    ret.append('input=%s' %input)
    ret.append('pool_size=%d' %pool_size)
    ret.append('strides=%d' %strides)
    ret.append('padding=%s' %padding)
    ret.append('')
    return ret


def AvgPool1D_proto(name,
                  input,
                  pool_size=2,
                  strides=2):
    ret = []
    ret.append('[AvgPool1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('')
    return ret


def AvgPool_proto(name,
                  input,
                  pool_size=2,
                  strides=2):
    ret = []
    ret.append('[AvgPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('')
    return ret


def GlobalAvgPool_proto(name,
                  input):
    ret = []
    ret.append('[GlobalAvgPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def GlobalAvgPool1D_proto(name,
                  input):
    ret = []
    ret.append('[GlobalAvgPool1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Input_proto(name,
          input_shape,
          dtype='float32',
          mean=-1,
          std=-1):
    ret = []
    ret.append('[Input]')
    ret.append('name=%s' % name)
    ret.append('input_shape=%s' % input_shape)
    ret.append('dtype=%s' % dtype)
    ret.append('mean=%s' %mean)
    ret.append('std=%s' %std)
    ret.append('')
    return ret


def Flatten_proto(name,
                  input):
    ret = []
    ret.append('[Flatten]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def MSE_Proto(name,
              logits,
              labels):
    ret = []
    ret.append('[MSE]')
    ret.append('name=%s' %name)
    ret.append('logits=%s' %logits)
    ret.append('labels=%s' %labels)
    ret.append('')

    return ret

def MAE_Proto(name,
              logits,
              labels):
    ret = []
    ret.append('[MAE]')
    ret.append('name=%s' %name)
    ret.append('logits=%s' %logits)
    ret.append('labels=%s' %labels)
    ret.append('')

    return ret


def Identity_proto(name,
                  input):
    ret = []
    ret.append('[Identity]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Output_proto(name,
                 input):
    ret = []
    ret.append('[Output]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Dropout_proto(name,
                input,
                dropout=0.0):
    ret = []
    ret.append('[Dropout]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('dropout=%s' % dropout)
    ret.append('')
    return ret

def Dropblock_proto(name,
                    input,
                    rate=0.0,
                    block_size=7):
    ret = []
    ret.append('[DropBlock]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('rate=%s' % rate)
    ret.append('block_size=%s' % block_size)
    ret.append('')
    return ret


def Dense_proto(name,
                input,
                units,
                dropout=0.0,
                activation='linear',
                batchnorm=True,
                regularizer_scale=1e-5,
                use_bias=True):
    ret = []
    ret.append('[Dense]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('units=%d' % units)
    ret.append('dropout=%s' % dropout)
    ret.append('activation=%s' % activation)
    ret.append('batchnorm=%s' % batchnorm)
    ret.append('regularizer_strength=%e' % regularizer_scale)
    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def Concat_proto(name,
                 input):
    ret = []
    ret.append('[Concat]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret

def Add_proto(name,
              input,
              activation,
              trainable=True):
    ret = []
    ret.append('[Add]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('trainable=%s' %trainable)
    ret.append('')
    return ret


def Add_n_proto(name,
                input,
                activation='linear'):
    ret = []
    ret.append('[Add_n]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' %activation)
    ret.append('')
    return ret


def Accuracy_proto(name,
                   logits,
                   labels):
    ret = []
    ret.append('[Accuracy]')
    ret.append('name=%s' % name)
    ret.append('logits=%s' % logits)
    ret.append('labels=%s' % labels)
    ret.append('')
    return ret

def BinaryAccuracy_proto(name,
                   logits,
                   labels):
    ret = []
    ret.append('[BinaryAccuracy]')
    ret.append('name=%s' % name)
    ret.append('logits=%s' % logits)
    ret.append('labels=%s' % labels)
    ret.append('')
    return ret

def SmoothedL1_proto(name,
                     input,
                     labels,
                     delta=1.0):
    ret = []
    ret.append('[SmoothedL1Loss]')
    ret.append('name=%s' %name)
    ret.append('input=%s' %input)
    ret.append('labels=%s' %labels)
    ret.append('delta=%s' %delta)
    ret.append('')
    return ret


def BCELoss_proto(name,
                  input,
                  labels):
    ret = []
    ret.append('[BCELoss]')
    ret.append('name=%s' %name)
    ret.append('input=%s' %input)
    ret.append('labels=%s' %labels)
    ret.append('')
    return ret

def Softmax_proto(name,
                  input,
                  labels,
                  label_smoothing=0.0):
    ret = []
    ret.append('[SoftmaxLoss]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('labels=%s' % labels)
    ret.append("label_smoothing=%s" %label_smoothing)
    ret.append('')
    return ret

def SoftmaxWithAuxTower_proto(name,
                              input,
                              labels,
                              label_smoothing=0.0,
                              aux_weight=.4,
                              aux_head=None,
                              aux_width=768,
                              aux_activation='relu'):
    ret = []
    ret.append('[SoftmaxLossWithAuxTower]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('labels=%s' % labels)
    ret.append("label_smoothing=%s" %label_smoothing)
    ret.append("aux_weight=%s" %aux_weight)
    ret.append("aux_head=%s" %aux_head)
    ret.append("aux_width=%s" %aux_width)
    ret.append("aux_activation=%s" %aux_activation)
    ret.append('')
    return ret

def TopkAcc_proto(name,
                  logits,
                  labels,
                  k):
    ret = []
    ret.append('[TopkAcc]')
    ret.append('name=%s' % name)
    ret.append('logits=%s' % logits)
    ret.append('labels=%s' % labels)
    ret.append('k=%d' % k)
    ret.append('')
    return ret

def YOLO_loss_proto(name,
                    input,
                    labels,
                    num_objects,
                    cell_size=7,
                    classes=20,
                    nms=0.35,
                    anchors=None,
                    boxes=2,
                    class_coef=1.0,
                    loc_coef=5.0,
                    obj_coef=1.0,
                    bg_coef=0.5,
                    image_size=224,
                    batch_size=64):
    ret = []
    ret.append('[YoloLoss]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('labels=%s' % labels)
    ret.append('num_objects=%s' % num_objects)
    ret.append('cell_size=%s' % cell_size)
    ret.append('classes=%s' % classes)
    ret.append('nms=%s' % nms)
    ret.append('anchors=%s' % anchors)
    ret.append('boxes=%s' % boxes)
    ret.append("class_coef=%s" % class_coef)
    ret.append("loc_coef=%s" % loc_coef)
    ret.append("obj_coef=%s" % obj_coef)
    ret.append("bg_coef=%s" % bg_coef)
    ret.append('image_size=%s' % image_size)
    ret.append('batch_size=%s' % batch_size)
    ret.append('')
    return ret


def Embedding_Proto(name,
                    input,
                    vocab_dim,
                    embedding_dim,
                    text_length):
    ret = []
    ret.append("[Embedding]")
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('vocab_dim=%s' %vocab_dim)
    ret.append('embedding_dim=%s' %embedding_dim)
    ret.append('text_length=%s' %text_length)
    ret.append('')
    return ret


def SE_Proto(name,
             input,
             se_ratio,
             trainable=True):
    """
    Squeeze-and-excitation layer prototxt.
    :param name:
    :param se_ratio:
    :return:
    """
    ret = []
    ret.append("[SE]")
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('se_ratio=%s' %se_ratio)
    ret.append('trainable=%s' %trainable)
    ret.append('')
    return ret

def ElasticSE_Proto(name,
                    input,
                    max_filters,
                    se_ratio,
                    trainable=True):
    """
    Squeeze-and-excitation layer prototxt.
    :param name:
    :param se_ratio:
    :return:
    """
    ret = []
    ret.append("[ElasticSE]")
    ret.append('name=%s' % name)
    ret.append('max_filters=%s' %max_filters)
    ret.append('input=%s' % input)
    ret.append('se_ratio=%s' %se_ratio)
    ret.append('trainable=%s' %trainable)
    ret.append('')
    return ret


