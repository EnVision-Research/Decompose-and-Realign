# Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models

[Luozhou Wang](https://wileewang.github.io/)$^{{\*}}$, [Guibao Shen]()$^{{\*}}$, [Wenhang Ge](https://g3956.github.io/wenhangge.github.io/), [Guangyong Chen](https://guangyongchen.github.io/), [Yijun Li](https://yijunmaverick.github.io/), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

HKUST(GZ), HKUST, ZJL, ZJU, Adobe.

${\*}$: Equal contribution.
\**: Corresponding author.

<a href="https://arxiv.org/abs/2306.14408"><img src="https://img.shields.io/badge/arXiv-2306.14408-b31b1b.svg" height=22.5></a>
<a href="https://a-bigbao.github.io/D-R/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<!--
<a href="https://youtu.be/9EWs2IX4cus"><img src="https://img.shields.io/static/v1?label=5-Minute&message=Video&color=darkgreen" height=20.5></a> 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/Attend-and-Excite)
[![Replicate](https://replicate.com/daanelson/attend-and-excite/badge)](https://replicate.com/daanelson/attend-and-excite)
-->

## 🎏 Abstract

Text-to-image diffusion models have advanced towards more controllable generation via supporting various additional conditions (e.g., depth map, bounding box) beyond text. However, these models are learned based on the premise of perfect alignment between the text and extra conditions.

<details><summary>CLICK for the full abstract</summary>

> Text-to-image diffusion models have advanced towards more controllable generation via supporting various additional conditions (e.g., depth map, bounding box) beyond text. However, these models are learned based on the premise of perfect alignment between the text and extra conditions. If this alignment is not satisfied, the final output could be either dominated by one condition, or ambiguity may arise, failing to meet user expectations. To address this issue, we present a training-free approach called "Decompose and Realign" to further improve the controllability of existing models when provided with partially aligned conditions. The "Decompose" phase separates conditions based on pair relationships, computing the result individually for each pair. This ensures that each pair no longer has conflicting conditions. The "Realign" phase aligns these independently calculated results via a cross-attention mechanism to avoid new conflicts when combining them back. Both qualitative and quantitative results demonstrate the effectiveness of our approach in handling unaligned conditions, which performs favorably against recent methods and more importantly adds flexibility to the controllable image generation process.

</details>

<div align=center>
<img src="resources/fig-teaser.png" width="97%"/>

llustration of our proposed Decompose and Realign showcasing the ability to handle the misalignment between conditions
for controllable generation task
</div>

## 🔧 Quick-Start
### Installation
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library. 

```
pip install diffusers
```

### Prepare your inputs
To generate an image using our model, structure the input conditions as a JSON object:
```json
{
    "text": {
        "caption": "A panda hails a taxi on the street with a red suitcase at its feet", 
        "index": [10,11,12], 
        "control_info": 10,
        "cfg":7
    }, 
    "pose": {
        "index": [1, 2], 
        "control_info": "resources/pose.png",
        "cfg":5
    }, 
    "bbox": {
        "index": [4, 5], 
        "control_info": [[0.1, 0.5, 0.6, 0.8]],
        "cfg":4
    }, 
    "depth": {
        "index": [6, 7, 8], 
        "control_info": "resources/depth.png",
        "cfg":2
    }
}
```
Notes:

- **Text**: Mandatory for generation. `index` specifies text tokens to enhance using our **Confidence Focusing Operation** and **Concentration Refinement Operation**, detailed in Sec 3.3 of our paper ([see code](https://github.com/wileewang/Decompose-and-Realign/blob/main/controller.py#L109-L110)). `control_info` acts as a multiplier for the attention values of these tokens, amplifying their visual prominence.

- **Image Conditions**: For keys such as `pose` and `depth`, we utilize [ControlNets](https://github.com/lllyasviel/ControlNet) which require a condition image. Here, `control_info` should be a path to the condition image. **Ensure all images are loaded as PIL.Image objects prior to their integration into the pipeline**.

- **Bounding Box (bbox)**: Implements control via a bounding box, in coordination with [grounding tokens](https://github.com/gligen/GLIGEN). The `control_info` for `bbox` should be formatted as `[x,y,w,h]`, with each value ranging from 0 to 1, representing the coordinates and dimensions of the bounding box.

- **Configuration Weights (cfg)**: Each control signal is assigned a `cfg` value, acting as a weight in the final composition process.

### Run
You can use our pipeline similarly to the [StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.enable_attention_slicing.example). Below is an example usage:

```python
import torch
from PIL import Image
from diffusers import ControlNetModel
from pipeline_decompose_and_realign import *

device = torch.device("cuda")

# Load required ControlNet models
controlnet_dict = {
    'depth': ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
    'pose': ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose"),
}

# Initialize and configure the pipeline
pipe = DecomposeAndRealignPipeline.from_pretrained("masterful/gligen-1-4-generation-text-box").to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_controlnet(controlnet_dict)

# Generate the output
output = pipe(
    inputs=data, # json object.
    negative_prompt='unnatural colors, bad proportions, worst quality',
    dr_scheduled_sampling_beta=0.5,
    generator=torch.Generator(device="cuda").manual_seed(20),
)
output.images[0].save('output.png')
```

Notes
- **ControlNet Integration**: Load and organize the required ControlNets into a dictionary, then register them to the pipeline using `pipe.set_controlnet(controlnet_dict)`.
- **Model Loading**: The adapter modules for [grounding tokens](https://github.com/gligen/GLIGEN) are integrated into the `masterful/gligen-1-4-generation-text-box` model, which can be directly loaded.
- **Parameter Setting**: The `dr_scheduled_sampling_beta` parameter controls the influence range of our method. A recommended setting is 0.5.


<div align=center>
<img src="resources/fig-teaser.png" width="97%"/>

llustration of our proposed Decompose and Realign showcasing the ability to handle the misalignment between conditions
for controllable generation task
</div>

## 🚧 Todo

- [x] Release the inference code
- [x] Release the guidance documents
- [ ] Release the gradio demo
- [ ] Release the extensions for Stable Diffusion WebUI


## 📍 Citation 
```
@misc{wang2023decompose,
      title={Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models}, 
      author={Luozhou Wang and Guibao Shen and Yijun Li and Ying-cong Chen},
      year={2023},
      eprint={2306.14408},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements 
This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library as well as the [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) codebase.

