# Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models

[Luozhou Wang](https://wileewang.github.io/)$^{{\*}}$, [Guibao Shen]()$^{{\*}}$, [Wenhang Ge](https://g3956.github.io/wenhangge.github.io/), [Guangyong Chen](https://guangyongchen.github.io/), [Yijun Li](https://yijunmaverick.github.io/), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

${\*}$: Equal contribution.
\**: Corresponding author.

[Paper PDF (Arxiv)](https://arxiv.org/abs/2306.14408) | [Project Page (Coming Soon)]() | [Gradio Demo](resources/Gradio_Demo.md)

---

<div align=center>
<img src="resources/fig-teaser.png" width="47.5%"/><img src="resources/fig-exp-multiple_conditions-1.png" width="47.5%"/>  

Note: we compress these motion pictures for faster previewing.
</div>

<div align=center>
<img src="resources/teaser.jpg" width="95%"/>  
  
Examples of text-to-3D content creations with our framework, the *LucidDreamer*, within **~35mins** on A100.
</div>


## 🎏 Abstract

Text-to-image diffusion models have advanced towards more controllable generation via supporting various additional conditions (e.g., depth map, bounding box) beyond text. However, these models are learned based on the premise of perfect alignment between the text and extra conditions.

<details><summary>CLICK for the full abstract</summary>

> Text-to-image diffusion models have advanced towards more controllable generation via supporting various additional conditions (e.g., depth map, bounding box) beyond text. However, these models are learned based on the premise of perfect alignment between the text and extra conditions. If this alignment is not satisfied, the final output could be either dominated by one condition, or ambiguity may arise, failing to meet user expectations. To address this issue, we present a training-free approach called "Decompose and Realign" to further improve the controllability of existing models when provided with partially aligned conditions. The "Decompose" phase separates conditions based on pair relationships, computing the result individually for each pair. This ensures that each pair no longer has conflicting conditions. The "Realign" phase aligns these independently calculated results via a cross-attention mechanism to avoid new conflicts when combining them back. Both qualitative and quantitative results demonstrate the effectiveness of our approach in handling unaligned conditions, which performs favorably against recent methods and more importantly adds flexibility to the controllable image generation process.

</details>

## 🔧 Training Instructions

Our code is now released! Please refer to this [**link**](resources/Training_Instructions.md) for detailed training instructions.

## 🤗 Gradio Demo

We are currently building an online demo of LucidDreamer with Gradio, you can check it out by clicking this [link](https://huggingface.co/spaces/haodongli/LucidDreamer). It is still under development, and the service might not be available from time to time. 

## 🚧 Todo

- [x] Release the basic training codes
- [x] Release the guidance documents
- [ ] Release the training codes for more applications


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

## Acknowledgement
This work is built on many amazing research works and open-source projects:
- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)

Thanks for their excellent work and great contribution to 3D generation area.
