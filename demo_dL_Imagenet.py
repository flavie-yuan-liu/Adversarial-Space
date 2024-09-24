import argparse
import os
import torch
import torchvision.models as models
from torch.utils.data import random_split
from attacks.adil_greedy import ADIL
import numpy as np
from DS_ImageNet import DS_ImageNet
from imagenet_loading import load_ImageNet, dataset_split_by_class
import random
import torchmetrics
from robustbench.utils import load_model
from tqdm import tqdm


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input-mean)/std


def model_accuracy(dataset, model, device='cpu'):
    metric = torchmetrics.Accuracy()
    metric.to(device)
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model = model.to(device)
            pred = model(x).softmax(dim=-1)
            acc = metric(pred, y)
        acc = metric.compute()
    metric.reset()
    return acc


def dir_model(model_name='', norm='Linf'):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True, progress=False)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True, progress=False)
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
    elif model_name == 'robust_r18':
        model = load_model(model_name='Salman2020Do_R18', dataset='imagenet', threat_model=norm)
    elif model_name == 'robust_50_2':
        model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model=norm)
    elif model_name == 'robust_r50':
        model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model=norm)

    if model_name.find('robust')==-1:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    device = torch.device(args.gpu)

    norm = args.norm
    print(norm)

    # ------------------------------------------------------------------------
    # loading model (densenet, googlenet, inception, mobilenetv2, resnet, vgg)
    # ------------------------------------------------------------------------

    source_models_name = [name.lower() for name in args.source_models]
    print(source_models_name, norm)
    source_models = dir_models(models_name_list=source_models_name, norm=norm)

    target_model_name = args.model.lower()
    target_model = dir_model(model_name=target_model_name, norm=norm)

    # ----------------------------------------------------------------------
    # loading imagenet data
    # ----------------------------------------------------------------------

    dataset, classes = load_ImageNet()
    acc = model_accuracy(dataset, target_model, device=device)
    print("accuracy of the the model {} is {}".format(target_model_name, acc*100))

    # Set the number of samples for training
    num_train_per_class = args.num_per_class  # set the number of samples for training 10 x number of classes
    num_val_per_class = 5
    num_test_per_class = 5

    # prepare the class-balanced dataset
    # default setting: 20 per class for training, 2 per class for validation, 5 per class for testing

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class],
                                 number_of_classes=args.number_of_class)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
    # print('test_dataset length', len(test_dataset))

    corr_sample = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        output = target_model.to(device)(x)
        corr_sample += torch.sum(y==output.argmax(dim=-1))
    acc = corr_sample/len(val_dataset)
    print(f"accuracy of {target_model_name} on ImageNet is {acc} ")
    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------
    train = args.train
    
    eps = args.eps/255 if norm.lower() == 'linf' else 0.5
    step_size = 2*eps
    step_size_v = 1. if norm.lower() == 'linf' else 0.1
    n_at = args.n_atom
    eps_coding = 10/255
    eps_set = [10/255, 12/255, 14/255, 16/255]
    step_size_v_train = args.step_size_v

    dict_models = ['resnet18', 'resnet50',  'densenet', 'vgg', 'robust_r18', 'robust_50_2'] 

    attack_adil = ADIL(target_model.to(device), steps=200,  batch_size=128, eps=eps, eps_coding=eps_coding,
                          step_size=step_size, loss='logits', norm=norm.lower(), model_name=target_model_name,
                       kappa=args.kappa)

    print('Adil Attack: step size, lambda, number of atoms, eps, eps_coding',
          [target_model_name, step_size, n_at, norm, int(eps * 255), int(eps_coding*255)])

    # dictionary training process
    if train:
        print('============begin training D===========')
        for source_model in source_models:
            source_model.to(device)
        attack_adil.learn_dictionary_greedy(source_models, train_dataset, val_dataset, step_size_v=step_size_v_train,
                                            n_add=50, all_data=args.all_data, time_consuming=args.time_consuming)

    for eps_test in eps_set:
        print(eps_test)
        attack_adil.eps_coding = eps_test
        fooling_samples = 0
        stop_early = False
        num_benign = 0
        for image, label in tqdm(test_loader):
            image, label = image.to(device), label.to(device)
            pred = target_model(image)
            pred_labels = pred.argmax(dim=-1)
            ind = pred_labels == label
            num_benign += torch.sum(ind.int())
            if num_benign > 1000:
                attack_img, success_attack = attack_adil(image[ind][:-(num_benign-1000)],
                                                                  label[ind][:-(num_benign-1000)],
                                                                  data_name='imagenet',
                                                                  source_model_name=source_models_name,
                                                                  step_size=step_size_v,
                                                                  black_box=False, n_add=50)
                stop_early = True
            else:
                attack_img, success_attack = attack_adil(image[ind], label[ind], data_name='imagenet',
                                                                  source_model_name=source_models_name,
                                                                  step_size=step_size_v,
                                                                  black_box=False, n_add=50)
            fooling_samples += success_attack
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
        required=True,
        default=[],
    )
    argparser.add_argument(
        '--seed', '-s',
        metavar='S',
        type=int,
        default=1,
        help='change seed to carry out the exp'
    )
    argparser.add_argument(
        '--number-of-class',
        type=int,
        default=1000,
        help='select number of class for training, default is 1000'
    )
    argparser.add_argument(
        '--step-size-v',
        type=float,
        default=1.,
        help='step size for updating v, the default value is 1.0'
    )
    argparser.add_argument(
        '--num-per-class',
        type=int,
        default=20,
        help='select number of class for training, default is 1'
    )
    argparser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='select number of class for training, default is 0'
    )
    argparser.add_argument(
        '--alpha',
        type=int,
        default=0,
        help='set tolerance of budget for training, default is 0'
    )
    argparser.add_argument(
        '--kappa',
        type=int,
        default=0,
        help='set the minimum limit for logit loss, default is 0'
    )
    argparser.add_argument(
        '--loss','-l',
        metavar='L',
        default='logits',
        help='choose the loss for attack learning, default is logits'
    )
    argparser.add_argument(
        '--method',
        default='gd',
        help='choose the optimisation method, default is gd (gradient descent), or you can change to alternating'
    )
    argparser.add_argument(
        '--norm',
        type=str,
        default='Linf',
        help='choose the distance measure used for attacking, the default is L2'
    )
    argparser.add_argument(
        '--eps',
        type=int,
        default=10,
        help='set the eps value, default is 10'
    )
    argparser.add_argument(
        '--step-size',
        type=float,
        default=0.001,
        help='set the step size for training dictionary, default is 0.001'
    )
    argparser.add_argument(
        '--transfer',
        type=bool,
        default=False,
        help='if the transferability performance is evaluated, the default value is True'
    )
    argparser.add_argument(
        '--train',
        type=bool,
        default=False,
        help='if the mode of the model is train, the default value is False'
    )
    argparser.add_argument(
        '--all-data',
        type=bool,
        default=False,
        help='if all data is used for training the dictionary, the default value is False'
    )
    argparser.add_argument(
        '--time-consuming',
        type=bool,
        default=False,
        help='if the time consuming mode is on where the validation process is active, the default value is False'
    )

    args = argparser.parse_args()
    models_name = ['resnet18', 'resnet50', 'densenet', 'vgg', 'robust_r18', 'robust_r50', 'robust_50_2']

    seed = args.seed
    print('seed:', seed, 'source_models', args.source_models, 'target_model', args.model, 'trainset', args.num_per_class*1000, 'step_size_v', args.step_size_v)
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    main(args)
