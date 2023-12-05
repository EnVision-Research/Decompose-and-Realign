import torch
from utils.attn_utils import *
from diffusers import ControlNetModel
from pipeline_decompose_and_realign import *
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

device = torch.device("cuda")
controlnet_dict = {
    'depth': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-depth'),
    'normal': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-normal'),
    'canny': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-canny'),
    'pose': ControlNetModel.from_pretrained('/home/wangluozhou/pretrained_models/sd-controlnet-pose'),
}

pipe = DecomposeAndRealignPipeline.from_pretrained('/home/wangluozhou/pretrained_models/gligen-1-4-generation-text-box').to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_controlnet(controlnet_dict)

dataset_path = '/home/wangluozhou/projects/Decompose-and-Realign/data/metadata_complex.jsonl'
dataset = load_jsonl(dataset_path)
base_dir = os.path.dirname(dataset_path)
for data in dataset:
    for key, value in data.items():
        if key!='text' and key!='bbox':
            value['control_info'] = Image.open(os.path.join(base_dir, value['control_info'],f'{key}.png')).convert('RGB')

data = dataset[0]

guidance_scale_dict = {
    'text':7,
    'bbox':3,
    'pose':3,
    'depth':3,
}
for key in data:
    data[key]['cfg'] = guidance_scale_dict[key]


data = sort_inputs(data)
output = pipe(
    inputs=data,
    negative_prompt='unnatural colors, bad proportions, worst quality',
    dr_scheduled_sampling_beta=0.5,
    height=512,
    width=512,
    generator=torch.Generator(device="cuda").manual_seed(3),
)
output.images[0]





