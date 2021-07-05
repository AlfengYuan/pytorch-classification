import argparse
import torch
from Auto_Json import run
import models
model_names = models.__all__

parser = argparse.ArgumentParser(description='PyTorch Auto generate json for tensorrtCV!')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-c', '--checkpoints', dest='checkpoints', default=None, type=str, 
                    help='load checkpoints to evaluate')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.pretrained:
        print("=> using pre-trained model '{}', checkpoints '{}'".format(args.arch, args.checkpoints if args.checkpoints is not None else "torchvision"))
        #model = models.__dict__[args.arch](pretrained=True)
        model = model_names[args.arch](pretrained=True, checkpoints=args.checkpoints)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch]()
        model = model_names[args.arch]()

    model.eval()
    x = torch.ones([1, 3, 224, 224])
    run(model.to("cuda:0"), x.to("cuda:0"))