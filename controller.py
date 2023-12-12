import torch
import math
from diffusers.models.attention_processor import Attention
import copy


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

class DRAttnController:

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        for _, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()

    def process_text_attention(self, attn, start_index, place_in_unet, is_cross):
        # whether apply up-only for attention of other control type, exclusive for controlnets.
        if self.cur_att_layer  == start_index:
            for key, value in self.attn_store_dicts.items():
                if key != 'bbox' and key != 'text':
                    value['attn_store']['down_cross'] = [
                        value['attn_store']['up_cross'][4], 
                        value['attn_store']['up_cross'][3], 
                        value['attn_store']['up_cross'][1], 
                        value['attn_store']['up_cross'][0]
                    ]

        # apply the realign
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            h = attn.shape[0]
            if is_cross:
                for element_key, element_value in self.attn_store_dicts.items():
                    if element_key!='text':
                        attn[h // 2:, :, element_value['index']] = element_value['attn_store'][key].pop(0)[:,:,1:-1]

                for element_key, element_value in self.attn_store_dicts.items():
                    if element_key == 'text':
                        if len(element_value['index'])!=0:
                            attn[h // 2:, :, element_value['index']] = self.max_op(attn[h // 2:, :, element_value['index']])
                            attn[h // 2:, :, element_value['index']] = self.blur_op(attn[h // 2:, :, element_value['index']])
                            attn[h // 2:, :, element_value['index']] *= element_value['control_info']
                
                # renormalize eot
                if self.renormalize:
                    sum_except_eot = torch.sum(attn[h // 2:, :, 0:-1], dim=2, keepdim=True)
                    attn[h//2:, :,  -1:] = 1 - sum_except_eot 
                    attn[h//2:, :,  :] = attn[h//2:,:,:].clamp(0,1)
            else:
                pass

    def store_attention(self, attn, attn_key, place_in_unet, is_cross):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            if is_cross:
                self.attn_store_dicts[attn_key]['attn_store'][key].append(attn)

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        dr_activated = True if self.cur_step < self.inf_step else False

        if dr_activated:
            for order, attn_key in enumerate(self.attn_store_dicts):
                # note to keep the order the same
                start_index = order * self.num_att_layers
                end_index = start_index + self.num_att_layers
                # when computing the attention of text and ensure the text is always the last one to compute
                if start_index <= self.cur_att_layer < end_index and attn_key == 'text':
                    if attn_key =='text':
                        self.process_text_attention(attn, start_index, place_in_unet, is_cross)
                    else:
                        self.store_attention(attn, attn_key, place_in_unet, is_cross)

            self.cur_att_layer += 1

            if self.cur_att_layer == self.num_att_layers + self.num_att_layers * (len(self.attn_store_dicts) - 1):
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()

        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        for _, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()
        
    def __init__(
            self, 
            inputs={},
            renormalize=False, 
            inf_step=50,
        ):
        self.inf_step = inf_step
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.curr_step_index = 0


        self.attn_store_dicts = copy.deepcopy(inputs)
        for _, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()

        self.renormalize = renormalize
        self.blur_op = GaussianBlurAttention(sigma=0.5)
        self.max_op = MaximumAttentionRegions(k_or_percentage=0.2)