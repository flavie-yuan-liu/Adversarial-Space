import argparse
import os
import time
from torch.utils.data import Subset
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import random_split
from attacks.adil_greedy import ADIL
from attacks.uap import UAP
from attacks.naa import NAA
from attacks.rap import RAP
from attacks.uappgd import UAPPGD
from attacks.soa import SOA
from cifar10_models import vgg, mobilenetv2, densenet, resnet, googlenet, inception
import numpy as np
import random
from torchvision.transforms import transforms
from robustbench.utils import load_model
from torchattacks import *
# from torchvision import models
import gradient_observation as graob


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input-mean)/std


class Subset_I(Subset):
    def __init__(self, dataset, indices, indexed=False):
        super(Subset_I, self).__init__(dataset=dataset, indices=indices)
        self.indexed = indexed

    def __getitems__(self, indices):
        if self.indexed:
            return [(idx, self.dataset[self.indices[idx]]) for idx in indices]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]


def dataset_split_by_class(dataset, number_per_class, number_of_classes):
    labels = dataset.targets
    sorted_idx = np.argsort(labels)
    num_classes = len(dataset.classes)
    matrix_sorted_idx = sorted_idx.reshape((num_classes, -1))

    split1 = number_per_class[0] + number_per_class[1]
    split2 = sum(number_per_class)

    for i in range(matrix_sorted_idx.shape[0]):
        random.shuffle(matrix_sorted_idx[i, :])

    indices_train = matrix_sorted_idx[:number_of_classes, 0:number_per_class[0]].flatten()
    indices_val = matrix_sorted_idx[:number_of_classes, number_per_class[0]:split1].flatten()
    indices_test = matrix_sorted_idx[:number_of_classes, split1:split2].flatten()

    return Subset_I(dataset, indices_train), Subset_I(dataset, indices_val), Subset_I(dataset, indices_test)


def dir_model(model_name='', norm='Linf'):

    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=True)
    elif model_name == 'densenet':
        model = densenet.densenet121(pretrained=True)
    elif model_name == 'googlenet':
        model = googlenet.googlenet(pretrained=True)
    elif model_name == 'mobilenet':
        model = mobilenetv2.mobilenet_v2(pretrained=True)
    elif model_name == 'inception':
        model = inception.inception_v3(pretrained=True)
    elif model_name == 'vgg':
        model = vgg.vgg11_bn(pretrained=True)
    elif model_name == 'robust_r18':
        model = load_model(model_name='Sehwag2021Proxy_R18', dataset='cifar10', threat_model=norm)
    elif model_name == 'robust_wrn_34_10':
        model = load_model(model_name='Sehwag2021Proxy', dataset='cifar10', threat_model=norm)
    elif model_name == 'robust_r152':
        model = load_model(model_name='Sehwag2021Proxy_ResNest152', dataset='cifar10', threat_model=norm)
    elif model_name == 'robust_r50':
        model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model=norm)

    if model_name.find('robust')==-1:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = torch.nn.Sequential(
            norm_layer,
            model
        )

    return model.eval()


def dir_models(models_name_list=[], norm='Linf'):
    assert not isinstance(models_name_list, str)
    models_set = []
    for model_name in models_name_list:
        models_set.append(dir_model(model_name, norm=norm))

    return models_set


def main(args):

    if not torch.cuda.is_available():
        print('Check cuda setting for model training on ImageNet')
        return

    if not args.distributed:
        torch.cuda.set_device(0)
        device = torch.device(0)
    else:
        device = 'cpu'

    norm = args.norm

    # =================================================================================================
    # ##########################        loading model        ##########################################
    # =================================================================================================
    source_models_name = [name.lower() for name in args.source_models]
    print(source_models_name, norm)
    source_models = dir_models(models_name_list=source_models_name, norm=norm)

    target_model_name = args.model.lower()
    target_model = dir_model(model_name=target_model_name, norm=norm)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # =================================================================================================
    # #########################        loading data         ###########################################
    # =================================================================================================
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Set the number of samples for training
    num_train_per_class = args.num_train_per_class  # set the number of samples for training 10 x number of classes
    num_val_per_class = 100
    num_test_per_class = 110

    # prepare the class-balanced dataset
    # default setting: 10 per class for training, 2 per class for validation, 5 per class for testing

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class],
                                 number_of_classes=args.trained_classes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False) #, pin_memory=True, num_workers=1)

    # ===================================================================================================
    # #########################       show models  accuracy         #######################################
    # ===================================================================================================
    def model_acc(model_name, model):
        corr_sample = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model.to(device)(x)
            corr_sample += torch.sum(y==output.argmax(dim=-1))
        acc = corr_sample/len(test_dataset)
        print(model_name, acc)

    model_acc(target_model_name, target_model)
    [model_acc(source_model_name, source_model) for source_model_name, source_model in zip(source_models_name, source_models)]

    # ====================================================================================================
    # ##########################     hyper-parameters setting    ########################################
    # ====================================================================================================

    # performance setting to evaluate
    train = args.train

    # grid searching of hyper-parameters

    eps = 8/255 if norm.lower() == 'linf' else 0.5
    step_size = 2*eps

    step_size_v = args.lr_inference if norm.lower() == 'linf' else 0.1
    eps_set = [8/255]
    bbo_models = 'cma'  # 'nes' is not sufficient to reaching a solution

    # parameters for transferability estimation
    dict_models = ['mobilenet', 'inception', 'googlenet', 'resnet', 'densenet', 'vgg', 'robust_r18', 'robust_wrn_34_10']

    attack_adil = ADIL(target_model.to(device), steps=200, batch_size=100, eps=eps, eps_coding=eps,
                          step_size=step_size, loss='logits', norm=norm.lower(), model_name=target_model_name,
                          bbo_methods=bbo_models, kappa=100)
    print('Adil Attack: step size, lambda, number of atoms, eps',
          [target_model_name, step_size, norm, int(eps * 255)])

    # dictionary training process
    if train:
        attack_adil.learn_dictionary_greedy(source_models, train_dataset, val_dataset, n_add=10)

    for eps_test in eps_set:

        attack_adil.eps = eps_test
        fooling_samples = 0
        stop_early = False
        mat_label = torch.tensor([])
        fr_vs_iter_all = None
        num_benign = 0
        for image, label in tqdm(test_loader):
            image, label = image.to(device), label.to(device)
            pred = target_model(image)
            pred_labels = pred.argmax(dim=-1)
            ind = pred_labels == label
            num_benign += torch.sum(ind.int())
            mat_label = torch.cat([mat_label, label[ind].detach().cpu()], dim=0)
            if num_benign > 1000:
                attack_img, success_attack, fr_vs_iter = attack_adil(image[ind][:-(num_benign-1000)],
                                                                  label[ind][:-(num_benign-1000)],
                                                                  data_name='cifar',
                                                                  source_model_name=source_models_name,
                                                                  source_models=source_models,
                                                                  step_size=step_size_v, eps_train=int(eps*255),
                                                                  black_box=False, n_add=args.n_add)
                stop_early = True
            else:
                attack_img, success_attack, fr_vs_iter = attack_adil(image[ind], label[ind], data_name='cifar',
                                                                  source_model_name=source_models_name,
                                                                  source_models=source_models,
                                                                  step_size=step_size_v, eps_train=int(eps*255),
                                                                  black_box=False, n_add=args.n_add)
            fooling_samples += success_attack
            fr_vs_iter_all = fr_vs_iter_all+fr_vs_iter if fr_vs_iter_all is not None else fr_vs_iter
            if stop_early or num_benign == 1000:
                break

        fooling_rate_auto = fooling_samples/1000
        print(target_model_name)
        print(f'adil fooling-rate: {fooling_rate_auto}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='vgg',
    )
    argparser.add_argument(
        '--source-models',
        action='append',
        help='Input models for training dictionary',
        default=['vgg'],
    )
    argparser.add_argument(
        '--seed', '-s',
        metavar='S',
        type=int,
        default=1,
        help='change seed to carry out the exp'
    )
    argparser.add_argument(
        '--num-train-per-class',
        type=int,
        default=100,
        help='number per class for training'
    )
    argparser.add_argument(
        '--trained-classes',
        metavar='TC',
        type=int,
        default=10,
        help='number of class for training'
    )
    argparser.add_argument(
        '--distributed',
        metavar='D',
        type=bool,
        default=False,
        help='If distributed data parallel used, default value is False'
    )
    argparser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='select number of class for training, default is 0'
    )
    argparser.add_argument(
        '--norm',
        type=str,
        default='Linf',
        help='choose the distance measure used for attacking, the default value is "Linf"'
    )
    argparser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate of dictionary learning'
    )
    argparser.add_argument(
        '--train',
        type=bool,
        default=False,
        help='if the mode of the model is train, the default value is False'
    )
    argparser.add_argument(
        '--lr-inference',
        type=int,
        default=1,
        help='learning rate for inference, default is 1'
    )
    argparser.add_argument(
        '--n-add',
        type=int,
        default=50,
        help='select coefficient of lasso regularization, default is 1'
    )
    args = argparser.parse_args()
    models_name = ['resnet18', 'densenet', 'vgg']
    # models_name = ['inception', 'resnet18', 'densenet', 'vgg', 'robust_r18', 'robust_wrn_34_10']

    seed = args.seed
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for n_add in [50]:
        print('seed:', seed, 'source_models', args.source_models, 'target_model', args.model)
        args.n_add = n_add
        print(args.train)
        main(args)
