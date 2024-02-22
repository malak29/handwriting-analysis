import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastai.vision.all import *
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.optimizer import SGD

def load_data(url):
    path = untar_data(url)
    return path

def explore_data(path):
    print(path.ls())
    train_path = path/'train'
    print(train_path.ls())

def display_sample_images(path, digit='3', n_images=5):
    digit_path = (path/'train'/digit).ls().sorted()
    for i in range(n_images):
        img = Image.open(digit_path[i])
        print(img)

def tensor_images(path, digit='3'):
    digit_tensors = [tensor(Image.open(o)) for o in (path/'train'/digit).ls()]
    stacked_digit_tensors = torch.stack(digit_tensors).float()/255
    return stacked_digit_tensors

def calculate_mean_image(stacked_tensors):
    return stacked_tensors.mean(0)

def show_mean_image(mean_image):
    show_image(mean_image)

def create_datasets(path):
    threes_tensors = tensor_images(path, '3')
    sevens_tensors = tensor_images(path, '7')
    train_x = torch.cat([threes_tensors, sevens_tensors]).view(-1, 28*28)
    train_y = tensor([1]*len(threes_tensors) + [0]*len(sevens_tensors)).unsqueeze(1)
    dset = list(zip(train_x, train_y))
    return dset

def create_dataloaders(dset, batch_size=256):
    dl = DataLoader(dset, batch_size=batch_size, shuffle=True)
    return dl

def linear_model(xb, weights, bias):
    return xb@weights + bias

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def train_epoch(model, lr, params, dl):
    for xb, yb in dl:
        calc_grad(xb, yb, model, params)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

def calc_grad(xb, yb, model, params):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

def validate_epoch(model, valid_dl):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

def train_model(model, epochs, lr, params, dl, valid_dl):
    for i in range(epochs):
        train_epoch(model, lr, params, dl)
        print(validate_epoch(model, valid_dl), end=' ')

# Example usage
path = load_data(URLs.MNIST_SAMPLE)
explore_data(path)
display_sample_images(path, '3', 5)
dset = create_datasets(path)
dl = create_dataloaders(dset)
weights = init_params((28*28, 1))
bias = init_params(1)
params = weights, bias
valid_dset = create_datasets(path/'valid')
valid_dl = create_dataloaders(valid_dset)
train_model(linear_model, 10, 1.0, params, dl, valid_dl)
