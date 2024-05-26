#!/usr/bin/env bash

#-------------transformer architecture-----------------------
#python main.py --method original --model transformer --dataset wikiText2 --B 128 --lr 0.01 --T 200 --device cuda:1 &
#python main.py --method low_rank --model transformer --dataset wikiText2 --B 128 --lr 0.01 --T 200 --device cuda:2 &
#python main.py --method low_rank --model transformer --dataset wikiText2 --B 128 --lr 0.01 --T 200 --regularization frobenius --coef_decay 1e-2 --device cuda:3 &


#--------------vision transformer------------------------------
#python main.py --method low_rank --model vit --dataset cifar10 --B 128 --optim Adam --T 500 --lr 0.001 --device cuda:2 &
#python main.py --method original --model vit --dataset cifar10 --B 128 --optim Adam --T 500 --lr 0.001 --device cuda:1
#python main.py --method low_rank --model vit --dataset cifar10 --B 128 --optim SGD --T 400 --lr 0.01 --regularization frobenius --coef_decay 1e-6 --device cuda:3 &
#python main.py --method low_rank --model vit --dataset cifar10 --B 128 --optim SGD --T 400 --lr 0.01 --regularization none --device cuda:2 &


#---------------vgg11-----------------------------------------
#python main.py --method original --model vgg11 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --device cuda:5 &
#python main.py --method low_rank --model vgg11 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --regularization frobenius --coef_decay 1e-2 --device cuda:0 &
#python main.py --method low_rank --model vgg11 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --regularization none --device cuda:4 &

#---------------vgg16-----------------------------------------
#python main.py --method original --model vgg16 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --device cuda:1 &
#python main.py --method low_rank --model vgg16 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --regularization frobenius --coef_decay 1e-2 --device cuda:1 &
#python main.py --method low_rank --model vgg16 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.01 --regularization none --device cuda:0 &

#-------------------resnet18---------------------------------
#python main.py --method original --model resnet18 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --device cuda:5 &
#python main.py --method low_rank --model resnet18 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --regularization frobenius --coef_decay 1e-4 --device cuda:1 &
#python main.py --method low_rank --model resnet18 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --regularization none --device cuda:5

#-------------------resnet101---------------------------------
#python main.py --method original --model resnet101 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --device cuda:1 &
#python main.py --method low_rank --model resnet101 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --regularization frobenius --coef_decay 1e-5 --device cuda:4 &
#python main.py --method low_rank --model resnet101 --dataset cifar10 --B 128 --optim SGD --T 200 --lr 0.1 --regularization none --device cuda:3 &

#--------------------LSTM-------------------------------------
#python main.py --method original --model lstm --dataset shakespeare --B 128 --optim SGD --T 100 --lr 0.01 --device cuda:3 &
#python main.py --method low_rank --model lstm --dataset shakespeare --B 128 --optim SGD --T 100 --lr 0.01 --device cuda:4