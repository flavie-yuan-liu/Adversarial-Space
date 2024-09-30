# Adversarial space

This implementation is based on the paper "The Data-Driven Transferable Adversarial Space", which focuses on identifying the embedded space where adversarial perturbations reside. This project specifically implements adversarial dictionary learning on two datasets: CIFAR10 and ImageNet.

To train the adversarial dictionary on CIFAR10 using a specific deep model, such as ResNet18, run the following command:

```
python3 demo_dl_cifar10.py --source-models resnet18 --train True
```
If you want to attack a specific deep model, such as VGG, you can specify the model as follows:
```
python3 demo_dl_cifar10.py --model vgg --source-models resnet18
```
For training on the ImageNet dataset, use the following command:
```
python3 demo_dL_Imagenet.py --source-models resnet18 --train True
```
If you prefer to skip training and just test the models, you can download the pre-trained adversarial spaces, on [CIFAR10 and IMAGENET](https://drive.google.com/drive/folders/1D46eqcv0Bs8L3yrMlAu7EbgVAXfFwkcQ?usp=sharing)


The trained adversarial dictionary has been shown to be model-agnostic, 


| |  | inception | resnet18 | ResNet50  | DenseNet | VGG | R-r18 | R-wrn-34-10 | R-r152 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |  -------- | 
|Inception | Ours  | 94.6 | 99.1 | 99.7 | 98.3 | 99.1 | 20.7 | 19.9 | 18.7 | 
|resnet18 | Ours   | 94.4 | 99.7 | 99.8 | 98.1 | 99.2 | 21.7 | 20.3 | 20.3 | 
|resnet50 | Ours   | 92.5 | 99.5  | 99.8 | 98.6 | 99.1 | 22.0 | 18.2 | 19.3 |
|densenet | Ours   | 94.5 | 99.7 | 99.8 | 98.9 | 99.1 | 22.6 | 20.8 | 19.3 |
|vgg | Ours | 93.5 | 99.5 | 99.5 | 98.5 | 99.4 | 20.8 | 19.8 | 19.9 |
|robust-r18 | Ours  | 92.1 | 99.1 | 99.8 | 98.2 | 99.2 | 31.7 | 27.6 | 25.8 |
|robust-wrn-34-10 | Ours  | 91.6 | 99.3 | 99.6 | 98.6 | 98.7 | 29.7 | 28.0 | 25.2 |
|robust-r152 | Ours  | 90.8 | 99.1 | 99.6 | 97.2 | 98.3 | 31.5 | 27.3 | 27.4 |
| reference attacks| AutoAttack | 100 | 100 | 100 | 100 | 100 | 35 | 31.5 | 28.6|
|  | PGD | 87.8 | 97.7 | 98.7 | 94.7 | 97.2 | 29.5 | 26.2 | 24.2 |

To generate an adversarial example using the learned adversarial dictionary, simply run:
```
python3 main_attack.py
```
This will output the results.
![attack_samples_dugong](https://github.com/flavie-yuan-liu/Adversarial-Space/blob/main/attack_samples_dugong.png)
![attack_samples_watch](https://github.com/flavie-yuan-liu/Adversarial-Space/blob/main/attack_samples_watch.png)
