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


## 🔧 Inference Instructions
### Environment
Our code builds on the requirement of the official [Stable Diffusion repository](https://github.com/CompVis/stable-diffusion). To set up their environment, please run:

```
conda env create -f environment/environment.yaml
conda activate ldm
```

On top of these requirements, we add several requirements which can be found in `environment/requirements.txt`. These requirements will be installed in the above command.

### Hugging Face Diffusers Library
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion v1.4 model. 


### Usage

<p align="center">
<img src="docs/results.jpg" width="800px"/>  
<br>
Example generations outputted by Stable Diffusion with Attend-and-Excite. 
</p>

To generate an image, you can simply run the `run.py` script. For example,
```
python run.py --prompt "a cat and a dog" --seeds [0] --token_indices [2,5]
```
Notes:

- To apply Attend-and-Excite on Stable Diffusion 2.1, specify: `--sd_2_1 True`
- You may run multiple seeds by passing a list of seeds. For example, `--seeds [0,1,2,3]`.
- If you do not provide a list of which token indices to alter using `--token_indices`, we will split the text according to the Stable Diffusion's tokenizer and display the index of each token. You will then be able to input which indices you wish to alter.
- If you wish to run the standard Stable Diffusion model without Attend-and-Excite, you can do so by passing `--run_standard_sd True`.
- All parameters are defined in `config.py` and are set to their defaults according to the official paper.

All generated images will be saved to the path `"{config.output_path}/{prompt}"`. We will also save a grid of all images (in the case of multiple seeds) under `config.output_path`.


<div align=center>
<img src="resources/fig-exp-multiple_conditions-1.png" width="97%"/>   
  
Examples of complex scenes, seamlessly integrating with a variety of controller mechanism.
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

