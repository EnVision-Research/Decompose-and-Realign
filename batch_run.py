from typing import List
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import pyrallis
import torch
import PIL
from PIL import Image
from pathlib import Path
from dataclasses import asdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from tqdm import tqdm
import os
import json
from pipeline_decompose_and_realign import *
# from pipeline_stable_diffusion_collective import *
from controlnet_aux import *
from diffusers import ControlNetModel
from PIL import Image, ImageDraw

from dataclasses import dataclass, field
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler



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


@dataclass
class Config:
    # negative prompt
    negative_prompt: str = "unnatural colors, bad proportions, worst quality"
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: range(100))
    # Path to load dataset
    dataset_path: str = '/home/wangluozhou/projects/Decompose-and-Realign/data/metadata_complex.jsonl'
    # Path to save all outputs to
    output_path: Path = Path('/home/wangluozhou/projects/Decompose-and-Realign/outputs/uncond/')
    # Number of denoising steps
    n_inference_steps: int = 50

    guidance_scale: dict = field(default_factory=lambda: {"text":7,"bbox":3,"depth":3,"normal":3,"canny":3,"pose":3})

    start_ratio: float = 0.0
    end_ratio: float = 1.0

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)



@pyrallis.wrap()
def main(config: Config):
    device = torch.device("cuda")
    pipe = DecomposeAndRealignPipeline.from_pretrained('/home/wangluozhou/pretrained_models/gligen-1-4-generation-text-box').to(device)
    controlnet_dict = {
        'depth': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-depth'),
        'normal': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-normal'),
        'canny': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-canny'),
        'pose': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-pose'),
    }
    pipe.set_controlnet(controlnet_dict)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    dataset = load_jsonl(config.dataset_path)
    base_dir = os.path.dirname(config.dataset_path)
    for data in dataset:
        for key, value in data.items():
            if key!='text' and key!='bbox':
                value['control_info'] = os.path.join(base_dir, value['control_info'])

    total_samples = len(dataset)
    start_index = int(total_samples * config.start_ratio)
    end_index = int(total_samples * config.end_ratio)

    for i in tqdm(range(start_index, end_index)):

        inputs = sort_inputs(dataset[i])

        controller = DRAttnController(inputs=inputs)
        register_attention_control(pipe, controller)

        for key in inputs:
            inputs[key]['cfg'] = config.guidance_scale[key]

        for seed in config.seeds:
            torch.cuda.empty_cache()
            g = torch.Generator('cuda').manual_seed(seed)
            controller.reset()
            outputs = pipe(
                inputs=inputs,
                negative_prompt=config.negative_prompt,
                height=512,
                width=512,
                boxes_scheduled_sampling_beta=1.0,
                generator=g
            )
            image = outputs.images[0]

            prompt_output_path = config.output_path / f'{str(i).zfill(3)}'
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}.png')

    # with open( config.output_path / f'{config.control_type}' / 'args.json', 'w') as f:
    #     json.dump(config.to_dict(), f, indent=4)

if __name__ == '__main__':
    main()
