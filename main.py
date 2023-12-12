import torch
from PIL import Image
from diffusers import ControlNetModel
from pipeline_decompose_and_realign import *

data = {
        "text": 
            {
                "caption": "A panda hails a taxi on the street with a red suitcase at its feet", 
                "index": [10,11,12], 
                "control_info": 10,
                "cfg":7,
            }, 
        "pose": 
            {
                "index": [1, 2], 
                "control_info": "resources/pose.png",
                "cfg":5
            }, 
        "bbox": 
            {
                "index": [4, 5], 
                "control_info": [[0.1, 0.5, 0.6, 0.8]],
                "cfg":4
            }, 
        "depth": 
            {
                "index": [6, 7, 8], 
                "control_info": "resources/depth.png",
                "cfg":2
            }
        }

for key, value in data.items():
    if key!='text' and key!='bbox':
        value['control_info'] = Image.open(value['control_info']).convert('RGB')


device = torch.device("cuda")

controlnet_dict = {
    'depth': ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
    'pose': ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose"),
}

pipe = DecomposeAndRealignPipeline.from_pretrained("masterful/gligen-1-4-generation-text-box").to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_controlnet(controlnet_dict)

output = pipe(
    inputs=data,
    negative_prompt='unnatural colors, bad proportions, worst quality',
    dr_scheduled_sampling_beta=0.5,
    generator=torch.Generator(device="cuda").manual_seed(20),
)
output.images[0].save('output.png')