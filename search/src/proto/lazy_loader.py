from src.proto.contrib import *

native_contrib_headers = \
    {# Full-precision headers
     "imagenet": lambda model_name: ImageNet_header(model_name),
     # Quantized model headers
     "imagenet-96-demo": lambda model_name: ImageNet_header_demo(model_name, 96),
     "imagenet-128-demo": lambda model_name: ImageNet_header_demo(model_name, 128),
     "imagenet-160-demo": lambda model_name: ImageNet_header_demo(model_name, 160),
     "imagenet-192-demo": lambda model_name: ImageNet_header_demo(model_name, 192),
     "imagenet-224-demo": lambda model_name: ImageNet_header_demo(model_name, 224),
     # Other headers
     "cifar10": lambda model_name: Cifar10_header(model_name),
     "cifar100": lambda model_name: Cifar100_header(model_name),
     "mnist": lambda model_name: MNIST_header(model_name),
     "eda-incomplete-net": lambda model_name: EDA_incomplete_net_header(model_name),
     'eda-density': lambda model_name: EDA_density_header(model_name)}

native_contrib_finals = \
    { # Full precision finals
     "imagenet": lambda out_name : ImageNet_final(out_name),
     "cifar10": lambda out_name: Cifar10_final(out_name),
     'cifar100': lambda out_name: Cifar100_final(out_name),
     "mnist": lambda out_name: MNIST_final(out_name),
     "eda-incomplete-net": lambda model_name: EDA_incomplete_net_final(model_name),
     'eda-density': lambda model_name: EDA_density_final(model_name)}

# Write the caffe headers after prototxt converter is done
caffe_headers = {"cifar10": lambda model_name: Caffe_Cifar10_header(model_name)}
caffe_finals = {"cifar10": lambda out_name: Caffe_Cifar10_final(out_name)}

headers = {"native": native_contrib_headers,
           "caffe": caffe_headers}

finals = {"native": native_contrib_finals,
          "caffe": caffe_finals}
