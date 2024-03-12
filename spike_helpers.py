import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
import numpy as np
import math

def getmatrix():
    data = np.load('./data.npz')
    Nd1 = np.tile(data['Nd'][:,:,np.newaxis],3)
    Nl1 = np.tile(data['Nl'][:,:,np.newaxis],3)
    D_light1 = np.tile(data['D_light'][:,:,np.newaxis],3)

    Nd2 = cv2.resize(Nd1, (400, 400), interpolation=cv2.INTER_LINEAR)
    Nl2 = cv2.resize(Nl1, (400, 400), interpolation=cv2.INTER_LINEAR)
    D_light2 = cv2.resize(D_light1, (400, 400), interpolation=cv2.INTER_LINEAR)

    return Nd1, Nl1, D_light1, Nd2, Nl2, D_light2

class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(1e-5).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 1e-5] = 0
        return grad_input

def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane  # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    out = out.detach() + (1 / threshold) * out - (1 / threshold) * out.detach()

    return membrane_potential, out

class SIFnode(nn.Module):
    def __init__(self):
        super(SIFnode,self).__init__()
    def forward1(self, x):
        membrane_potential = torch.zeros(x.shape[1:])
        results = list()
        for item in x:
            membrane_potential = membrane_potential + item
            membrane_potential, out = IF_Neuron(membrane_potential, 1.0)
            results.append(out)
        y = torch.sum(torch.stack(results, dim=0), dim=0) / 256
        y = torch.pow(y, 1 / 2.2)
        return y
    def forward2(self, x):
        membrane_potential = torch.zeros(x.shape[1:])
        results = list()
        for item in x:
            membrane_potential = membrane_potential + item
            membrane_potential, out = IF_Neuron(membrane_potential, 1.0)
            results.append(out)
        y = torch.sum(torch.stack(results, dim=0), dim=0) / 256
        y = torch.pow(y, 1 / 2.2)
        return y
    def noise(self,x,Nd,Nl,D_light):
        x = x/4
        D_true = 1 / (x + 1e-5) - 1
        D_fpn = D_true / (Nl + D_true * Nd / (D_light * (335 + Nd)))
        img_fpn = 1 / (D_fpn + 1)
        return img_fpn