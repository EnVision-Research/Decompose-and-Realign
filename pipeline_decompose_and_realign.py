import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import os
import PIL
import numpy as np
import torch
from torch.nn import functional as F
import copy

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention
from utils.attn_utils import MaximumAttentionRegions, GaussianBlurAttention

import sys
sys.path.append("../")
sys.path.append("../../")

logger = logging.get_logger(__name__)

class DRAttnController:

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        if self.viz_attn:
            for global_key, viz_key in zip(self.global_attention_store_dicts, self.viz_attn_store_dicts):
                if len(self.global_attn_store_dicts[global_key]['attn_store']) == 0:
                    self.global_attn_store_dicts[global_key]['attn_store'] = self.viz_attn_store_dicts[viz_key]['attn_store']
                else:
                    for key in self.global_attn_store_dicts[global_key]['attn_store']:
                        for i in range(len(self.global_attn_store_dicts['global_key']['attn_store'][key])):
                            self.global_attn_store_dicts[global_key]['attn_store'][key][i] += self.viz_attn_store_dicts[viz_key]['attn_store'][key][i]
                self.viz_attn_store_dicts[viz_key]['attn_store'] = self.get_empty_store()
        for key, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()

    def get_average_global_attention(self, key):   
        global_attention_store = self.global_attention_store_dicts[key]['attn_store']
        average_attention = {key: [item / self.cur_step for item in global_attention_store[key]] for key in global_attention_store}
        return average_attention

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        use_realign = True if self.cur_step < self.inf_step else False

        for order, (attn_key, viz_key) in enumerate(zip(self.attn_store_dicts, self.viz_attn_store_dicts)):
            # note to keep the order the same
            start_index = order * self.num_att_layers
            end_index = start_index + self.num_att_layers
            # when computing the attention of text and ensure the text is always the last one to compute
            if start_index <= self.cur_att_layer < end_index and attn_key == 'text':
                # whether apply up only for attention of other control type
                if self.cur_att_layer  == start_index and use_realign:
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
                        sum_except_text = 0
                        for element_key, element_value in self.attn_store_dicts.items():
                            if element_key!='text' and use_realign:
                                attn[h // 2:, :, element_value['index']] = element_value['attn_store'][key].pop(0)[:,:,1:-1]
                                sum_except_text += torch.sum(attn[h // 2:, :, element_value['index']], dim=2, keepdim=True)
                        
                        for element_key, element_value in self.attn_store_dicts.items():
                            if element_key == 'text':
                                if len(element_value['index'])!=0:
                                    attn[h // 2:, :, element_value['index']] = self.max_op(attn[h // 2:, :, element_value['index']])
                                    attn[h // 2:, :, element_value['index']] = self.blur_op(attn[h // 2:, :, element_value['index']])
                                    attn[h // 2:, :, element_value['index']] *= element_value['control_info']
                        
                        # renormalize eot
                        sum_except_eot = torch.sum(attn[h // 2:, :, 0:-1], dim=2, keepdim=True)
                        attn[h//2:, :,  -1:] = 1 - sum_except_eot 
                        attn[h//2:, :,  :] = attn[h//2:,:,:].clamp(0,1)
                    else:
                        pass
                    if self.viz_attn:
                        self.viz_attn_store_dicts[viz_key]['attn_store'][key].append(attn[h // 2:]) 

            elif start_index <= self.cur_att_layer < end_index and use_realign:
                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                if attn.shape[1] <= 32 ** 2:
                    if is_cross:
                        self.attn_store_dicts[attn_key]['attn_store'][key].append(attn)
                    if self.viz_attn:
                        self.viz_attn_store_dict[viz_key]['attn_store'][key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_att_layers * (len(self.attn_store_dicts) - 1):
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        for key, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()
        
    def __init__(
            self, 
            inputs={},
            renormalize=False, 
            viz_attn=False,
            inf_step=50,
        ):
        self.inf_step = inf_step
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.curr_step_index = 0

        self.viz_attn = viz_attn

        self.attn_store_dicts = copy.deepcopy(inputs)
        for key, value in self.attn_store_dicts.items():
            value['attn_store'] = self.get_empty_store()

        self.viz_attn_store_dicts = copy.deepcopy(self.attn_store_dicts)
        self.global_attention_store_dicts = copy.deepcopy(self.attn_store_dicts)
        self.renormalize = renormalize

        self.blur_op = GaussianBlurAttention(sigma=0.5)
        self.max_op = MaximumAttentionRegions(k_or_percentage=0.2)

class DummyAttnProcessor:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MyAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        attention_probs = self.attnstore(attention_probs, is_cross, self.place_in_unet) # 

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(model, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.endswith("fuser.attn.processor"):
            attn_procs[name] = DummyAttnProcessor()
            continue
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = MyAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

class DecomposeAndRealignPipeline(StableDiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers
    ):
        # super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def set_controlnet(self, controlnet_dict):
        for _, model in controlnet_dict.items():
            model = model.to(device=self._execution_device, dtype=self.unet.dtype)
        self.controlnet_dict = controlnet_dict

    def enable_fuser(self, enabled=True):
        for module in self.unet.modules():
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

    def crop(self, im, new_width, new_height):
        width, height = im.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return im.crop((left, top, right, bottom))

    def target_size_center_crop(self, im, new_hw):
        width, height = im.size
        if width != height:
            im = self.crop(im, min(height, width), min(height, width))
        return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=seq_len,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def process_input(
            self, 
            inputs, 
            device, 
            num_images_per_prompt, 
            negative_prompt, 
            batch_size,
            height,
            width
        ):
        for key, value in inputs.items():
            if key == 'text':
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt=value['caption'],
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            elif key == 'bbox':
                sub_caption = ' '.join(inputs['text']['caption'].split()[value['index'][0]-1:value['index'][-1]])
                prompt_embeds, _ = self.encode_prompt(
                    prompt=sub_caption,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                max_objs = 30
                if isinstance(sub_caption, str):
                    sub_caption = [sub_caption]

                gligen_phrases = sub_caption[:max_objs]
                gligen_boxes = value['control_info'][:max_objs]
                # prepare batched input to the PositionNet (boxes, phrases, mask)
                # Get tokens for phrases from pre-trained CLIPTokenizer
                tokenizer_inputs = self.tokenizer(gligen_phrases, padding=True, return_tensors="pt").to(device)
                # For the token, we use the same pre-trained text encoder
                # to obtain its text feature
                _text_embeddings = self.text_encoder(**tokenizer_inputs).pooler_output
                n_objs = len(gligen_boxes)
                # For each entity, described in phrases, is denoted with a bounding box,
                # we represent the location information as (xmin,ymin,xmax,ymax)
                boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
                boxes[:n_objs] = torch.tensor(gligen_boxes)
                text_embeddings = torch.zeros(
                    max_objs, self.unet.cross_attention_dim, device=device, dtype=self.text_encoder.dtype
                )
                text_embeddings[:n_objs] = _text_embeddings
                # Generate a mask for each object that is entity described by phrases
                masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
                masks[:n_objs] = 1

                repeat_batch = batch_size * num_images_per_prompt
                boxes = boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
                text_embeddings = text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
                masks = masks.unsqueeze(0).expand(repeat_batch, -1).clone()
                cross_attention_kwargs = {}
                cross_attention_kwargs["gligen"] = {"boxes": boxes, "positive_embeddings": text_embeddings, "masks": masks}
                value['cross_attention_kwargs'] = cross_attention_kwargs
            else:
                sub_caption = ' '.join(inputs['text']['caption'].split()[value['index'][0]-1:value['index'][-1]])
                prompt_embeds, _ = self.encode_prompt(
                    prompt=sub_caption,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                image = self.prepare_image(
                    image=value['control_info'],
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet_dict[key].dtype,
                    do_classifier_free_guidance=False,
                    # guess_mode=guess_mode,
                )
                value['control_info'] = image
            value['prompt_embeds'] = prompt_embeds
        return inputs
        
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self, 
        inputs = {},   
        dr_scheduled_sampling_beta: float = 0.5,
        height: Optional[int] = None, 
        width: Optional[int] = None, 
        num_inference_steps: int = 50, 
        negative_prompt: Optional[Union[str, List[str]]] = None, 
        num_images_per_prompt: Optional[int] = 1, 
        eta: float = 0, 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 
        latents: Optional[torch.FloatTensor] = None, 
        output_type: Optional[str] = "pil", 
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        torch.cuda.empty_cache()

        # regisiter the controller
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        
        inputs = self.process_input(
            copy.deepcopy(inputs), 
            device, 
            num_images_per_prompt, 
            negative_prompt, 
            batch_size,
            height,
            width
        )

        dtype = inputs["text"]['prompt_embeds'].dtype

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        num_dr_steps = int(dr_scheduled_sampling_beta * len(timesteps))
        controller = DRAttnController(
            inputs=inputs,
            inf_step=num_dr_steps
        )
        register_attention_control(self, controller)
        self.enable_fuser(True)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if i >= num_dr_steps:
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=inputs['text']['prompt_embeds'],
                    ).sample
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)

                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=0.7)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                else:
                    noise_pred_uncond = None
                    # compute each noise_pred for input in inputs:
                    for key, value in inputs.items():
                        if key == 'text':
                            latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=value['prompt_embeds'],
                            ).sample
                            noise_pred_text_uncond, noise_pred_text_cond = noise_pred.chunk(2)
                            value['noise_pred'] = noise_pred_text_cond
                            noise_pred_uncond = noise_pred_text_uncond

                        elif key == 'bbox' and value['cfg']!=0:
                            latent_model_input = latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=value['prompt_embeds'],
                                cross_attention_kwargs=value['cross_attention_kwargs'],
                            ).sample
                            value['noise_pred'] = noise_pred
                            
                        elif value['cfg']!=0:
                            latent_model_input = latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            down_block_res_samples, mid_block_res_sample = self.controlnet_dict[key](
                                latent_model_input,
                                t,
                                encoder_hidden_states=value['prompt_embeds'],
                                controlnet_cond=value['control_info'],
                                return_dict=False,
                            )

                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=value['prompt_embeds'],
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                            value['noise_pred'] = noise_pred
                
                    noise_pred = noise_pred_uncond.clone()

                    for key, value in inputs.items():
                        if value['cfg'] != 0:
                            noise_pred += value['cfg'] * (value['noise_pred'] - noise_pred_uncond)

                    noise_pred = rescale_noise_cfg(noise_pred, inputs['text']['noise_pred'], guidance_rescale=0.7)
                
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        controller.reset()
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

