# pytorch-classification


## trian (more reference [pytorch-examples-imagenet](https://github.com/pytorch/examples/tree/master/imagenet))
- python main.py -a alexnet --lr 0.01

## test
- python main.py -a alexnet -e --pretrained

## visual
- python visualization.py alexnet

## generate_json.py(Generate Json File for [tensorrtCV](https://github.com/wdhao/tensorrtCV))
- python generate_json.py -a alennet --pretrained

# model_zoo(imagenet dataset)
- batch_time is a reference value, not necessarily accurate, please use it with caution.
- batch_time(s/each 256 image) [One 1080Ti GPU]

| model | top1-acc | top5-acc | 1080Ti |
| --- | --- | --- | --- |
| alexnet | 56.630 | 79.054 | 0.488 |
| vgg11 | 68.872 | 88.658 | 0.521 |
| vgg11_bn | 70.408 | 89.724 | 0.513 |
| vgg13 | 69.984 | 89.306| 0.555 |
| vgg13_bn | 71.618 | 90.360 | 0.625 |
| vgg16 | 71.628 | 90.368 | 0.642 |
| vgg16_bn | 73.476 | 91.536 | 0.716 |
| vgg19 | 72.360 | 90.850 | 0.736 |
| vgg19_bn | 74.216 | 91.848 | 0.815 |
| googlenet | 69.744 | 89.544 | 0.504 |
| inception_v3(299 x 299) | 77.248 | 93.520 | 0.670 |
| resnet18 | 69.644 | 88.982 | 0.492 |
| resnet34 | 73.266 | 91.430 | 0.503 |
| resnet50 | 76.012 | 92.934 | 0.523 |
| resnet101 | 77.314 | 93.556 | 0.638 |
| resnet152 | 78.250 | 93.982 | 0.895 |
| resnext50_32x4d | 77.628 | 93.680 | 0.554 |
| resnext101_32x8d | 79.210 | 94.556 | 1.414 |
| wide_resnet50_2 | 78.464 | 94.064 | 0.678 |
| wide_resnet101_2 | 78.910 | 94.344 | 1.116 |
| densenet121 | 74.472 | 91.974 | 0.520 |
| densenet161 | 77.146 | 93.602 | 0.984 |
| densenet169 | 75.628 | 92.810 | 0.537 |
| densenet201 | 76.932 | 93.390 | 0.687 |
| squeezenet1_0 | 58.000 | 80.488 | 0.496 |
| squeezenet1_1 | 58.184 | 80.514 | 0.493 |
| shufflenet_v2_x0_5 | 60.646 | 81.696 | 0.488 |
| shufflenet_v2_x1_0 | 69.402 | 88.374 | 0.490 |
| mobilenet_v2 | 71.850 | 90.334 | 0.502 |
| mobilenetv3_small | 67.430 | 87.278 | 0.493 |
| mnasnet0_5 | 67.830 | 87.456 | 0.490 |
| mnasnet1_0 | 73.402 | 91.454 | 0.500 |
| efficientnet_b0 | 76.090 | 93.006 | 0.499 |
| efficientnet_b1(240 x 240) | 78.166 | 93.994 | 0.563 |
| efficientnet_b2(260 x 260) | 79.298 | 94.510 | 0.727 |
| efficientnet_b3(300 x 300) | 81.126 | 95.518 | - |
| hrnet_w18 | 76.832 | 93.404 | 0.617 |
| hrnet_w18_small_v1 | 72.276 | 90.586 | 0.499 |
| hrnet_w18_small_v2 | 75.164 | 92.430 | 0.506 |
| hrnet_w30 | 78.134 | 94.192 | 0.798 |
| ghostnet_1x | 73.938 | 91.470 | 0.495 |
| res2net_dla60 | 78.522 | 94.252 | 0.556 |
| res2next_dla60 | 78.322 | 94.190 | 0.582 |
| res2net50_v1b_26w_4s | 80.208 | 95.044 | 0.569 |
| res2net101_v1b_26w_4s | 81.196 | 95.392 | 0.890 |
| res2net50_26w_4s | 77.966 | 93.830 | 0.542 |
| res2net101_26w_4s | 79.120 | 94.432 | 0.851 |
| res2net50_26w_6s | 78.604 | 94.164 | 0.746 |
| res2net50_26w_8s | 79.130 | 94.404 | 0.934 |
| res2net50_48w_2s | 77.518 | 93.582 | 0.539 |
| res2net50_14w_8s | 78.118 | 93.836 | 0.602 |
| res2next50 | 78.080 | 93.952 | 0.590 |
| regnet_200M | 67.590 | 88.028 | 0.497 |
| regnet_400M | 71.946 | 90.632 | 0.493 |
| regnet_600M | 73.554 | 91.570 | 0.501 |
| regnet_800M | 74.880 | 92.294 | 0.503 |
| regnet_1600M | 76.996 | 93.452 | 0.510 |
| regnet_3200M | 78.358 | 94.162 | 0.523 |
| regnet_6400M | 79.202 | 94.764 | 0.659 |
| resnest50 | 80.970 | 95.350 | 1.074 |
| resnest50_fast_1s1x64d | 80.150 | 95.112 | 0.513 |
| resnest50_fast_2s1x64d | 80.472 | 95.262 | 0.554 |
| resnest50_fast_1s2x40d | 80.400 | 95.310 | 0.558 |
| resnest50_fast_2s2x40d | 80.626 | 95.412 | 0.561 |
| resnest50_fast_1s4x24d | 80.870 | 95.364 | 0.550 |















