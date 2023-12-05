import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.attention_processor import Attention
import torch
import math
import torch.nn.functional as F
import json

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # print(line)  # Add this line to print each line before parsing
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, filename): 
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def sort_inputs(dict):

    def custom_sort(item):
        key, _ = item
        return (key == 'text', key)
    
    sorted_items = sorted(dict.items(),key=custom_sort)
    sorted_dict = {k:v for k,v in sorted_items}

    return sorted_dict

class MaximumAttentionRegions:
    def __init__(self, k_or_percentage):
        self.k_or_percentage = k_or_percentage

    def __call__(self, attention):
        # attention shape [8,1024,seq_len]
        if 0 < self.k_or_percentage <= 1:  # Assume it's a percentage
            k = int(self.k_or_percentage * attention.size(1))
        else:  # Assume it's an absolute number
            k = self.k_or_percentage
        
        # Apply operation only to the specified tokens
        for idx in range(attention.shape[-1]):
            token_attention = attention[:, :, idx]
            top_values, _ = token_attention.topk(k, dim=1)
            threshold = top_values[:, -1].unsqueeze(1)
            attention[:, :, idx] = (token_attention >= threshold).float() * token_attention
        attention = attention.clamp(0, 1)
        return attention

class GaussianBlurAttention:
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Generate Gaussian kernel
        self.gaussian_kernel = self._get_gaussian_kernel(kernel_size, sigma)

    def _get_gaussian_kernel(self, kernel_size=3, sigma=1.0, channels=1):
        # Create a 2D Gaussian kernel
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def __call__(self, attention):
        #
        device = attention.device
        self.gaussian_kernel = self.gaussian_kernel.type_as(attention).to(device)
        # Only blur attention maps of specified tokens
        for idx in range(attention.shape[-1]):
            # Get the attention map corresponding to the token
            token_attention = attention[:, :, idx]
            spatial_dim = int((attention.size(1))**0.5)

            # Reshape it to apply 2D convolution (assumes spatial map is 16x16)
            token_attention = token_attention.view(token_attention.size(0), 1, spatial_dim, spatial_dim)

            # Apply the blur
            blurred_attention = F.conv2d(token_attention, self.gaussian_kernel, padding=self.kernel_size//2)
            
            # Reshape back and update the attention_probs tensor
            attention[:, :, idx] = blurred_attention.view(token_attention.size(0), -1)

        # Clamping the values to ensure they are within [0,1]
        attention = attention.clamp(0, 1)

        return attention