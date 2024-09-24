import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import random_split
from imagenet_loading import load_ImageNet
from DS_ImageNet import DS_ImageNet
from attacks.adil_greedy import ADIL
from PIL import Image
from torchvision.transforms import transforms
import torchattacks
import random


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input-mean)/std


def main(args):
    # check if gpu available
    if not torch.cuda.is_available():
        print('Check cuda setting for model training on ImageNet')
        return

    torch.cuda.set_device(0)
    device = torch.device(0)

    # ------------------------------------------------------------------------
    # loading model (densenet, googlenet, inception, mobilenetv2, resnet, vgg)
    # ------------------------------------------------------------------------
    model_name = args.model.lower()
    if model_name == 'resnet':
        model = models.resnet18(pretrained=True, progress=False)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True, progress=False)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True, progress=False)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, progress=False)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True, progress=False)
    elif model_name == 'vgg':
        model = models.vgg11(pretrained=True, progress=False)

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(
        norm_layer,
        model
    )
    model = model.eval()

    # ----------------------------------------------------------------------
    # loading image
    # ----------------------------------------------------------------------
    data, classes = load_ImageNet()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    adversarial_model_file = "/home/flavie/Desktop/dl-attacks/results_in_server/greedy_imagenet_v2_corr/1.0_1/adversarial_directions_vgg_full_50_atom_added_on_dataset_20000.bin"
    d = torch.load(adversarial_model_file)[0].data
    # im_path = '/home/flavie/Desktop/attack_learning/DL_attack/data/ImageNet/ILSVRC/Data/val/n04328186/ILSVRC2012_val_00025757.JPEG'
    im_path = '/home/flavie/Desktop/attack_learning/DL_attack/data/ImageNet/ILSVRC/Data/val/n02074367/ILSVRC2012_val_00040083.JPEG'

    im = Image.open(im_path)

    im = transform(im)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------
    eps = 4/255
    attack = ADIL(model.to(device), eps_coding=eps, model_name=model_name)

    im = im.to(device=device)
    y = 2
    label = model(im.unsqueeze(0)).argmax(dim=-1)
    adversary, _, _ = attack.coding(im.unsqueeze(0), label.unsqueeze(0), d, early_stop=True, loss_coding='logits')
    attack_label = model(adversary).argmax(dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # plt.axis('off')
    axes[0].imshow(im.detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[0].set_title(f'orginal image: {classes[label]}', fontsize=20)
    axes[0].set_axis_off()
    # plt.axis('off')
    scaled_pert = (adversary[0, :, :, :]-im+eps)/torch.max(adversary[0, :, :, :]-im+eps)
    axes[1].imshow(scaled_pert.detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[1].set_title(f'pertubation', fontsize=20)
    axes[1].set_axis_off()
    axes[2].imshow(adversary[0, :, :, :].detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[2].set_title(f'attack image: {classes[attack_label]}', fontsize=20)
    axes[2].set_axis_off()
    #fig.tight_layout(pad=0.5)
    plt.savefig('attack_samples.png')
    plt.show()

    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='resnet',
    )
    args = argparser.parse_args()
    main(args)
