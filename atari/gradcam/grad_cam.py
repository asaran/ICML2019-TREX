#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        # print('logits: ', type(self.logits), self.logits)
        self.probs = F.softmax(self.logits, dim=1)
        # indices = torch.tensor()
        # return self.probs, [list(range(len(self.probs[0])))]
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        # print('backprop device: ', self.device)
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                # self.fmap_pool[key] = output[len(output)-1].detach() # other activation maps also returned by model.forward()
                # print('output: ', output)
                self.fmap_pool[key] = output.detach()
            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        # print(gcam)
        gcam = F.relu(gcam)  # mostly gcam has negative values, everything converted to 0 
        # print(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )
        # print(gcam)
        B, C, H, W = gcam.shape # 32,1,84,84
        # print(B,C,H,W)
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        if torch.sum(gcam)==0.0:
            print("Warning! The GradCAM map is all zeros")
        epsilon = 1e-3
        max_tensor1 = gcam.max(dim=1, keepdim=True)[0]
        max_tensor2 = max_tensor1.clone()
        max_tensor2[max_tensor1==0.0] = epsilon
        # if sum()!=0.0: # else generates nan gracam maps
            # TODO: instead check if any max element is 0, or replace any 0 with epsilon
            # gcam /= gcam.max(dim=1, keepdim=True)[0]  # nans produced coz all values are 0, division by 0

        gcam /= max_tensor2
        gcam = gcam.view(B, C, H, W)
        # print(gcam)

        #TODO: set self.model.zero_grad()? so gradients not accumulated for forward pass
        self.model.zero_grad()
        return gcam