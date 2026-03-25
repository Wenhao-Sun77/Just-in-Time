# Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers (CVPR 2026)

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.10744-B31B1B.svg)](https://arxiv.org/abs/2603.10744)
[![Project](https://img.shields.io/badge/Project-Page-1F6FEB.svg)](https://wenhao-sun77.github.io/JiT/)
[![License](https://img.shields.io/badge/License-Apache_2.0-6E7781.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Repo stars](https://img.shields.io/github/stars/Wenhao-Sun77/Just-in-Time?style=plastic&logo=github)](https://github.com/Wenhao-Sun77/Just-in-Time)

</div>

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://wenhao-sun77.github.io/JiT/">
        <img src="./assets/A panda wearing a hoodie is looking at a blackboard with a rocket drawn on it, the blackboard reads “空间域加速”.png" alt="Showcase 1" width="100%">
      </a>
    </td>
    <td align="center" width="50%">
      <a href="https://wenhao-sun77.github.io/JiT/">
        <img src="./assets/A magical ancient book where glowing, golden runes forming the word 'Just in Time' are lifting off the page, mystical atmosphere..png" alt="Showcase 2" width="100%">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://wenhao-sun77.github.io/JiT/">
        <img src="./assets/A hyper-realistic portrait of a young woman in natural sunlight, soft shadows, shallow depth of field..png" alt="Showcase 3" width="100%">
      </a>
    </td>
    <td align="center" width="50%">
      <a href="https://wenhao-sun77.github.io/JiT/">
        <img src="./assets/A steaming cup of espresso on a rain-streaked window sill, soft morning light, hyper-realistic texture..png" alt="Showcase 4" width="100%">
      </a>
    </td>
  </tr>
</table>

## Introduction
**Just-in-Time (JiT)** is a training-free, model-agnostic acceleration framework for Diffusion Transformers (DiTs). By leveraging a Spatially Approximated Generative ODE (SAG-ODE) and a Deterministic Micro-Flow (DMF), JiT dynamically allocates computation to high-information-density spatial regions without requiring architectural changes, fine-tuning, or distillation.

### Key Features
- **Training-Free:** Plug-and-play acceleration for pre-trained DiTs. No distillation, no fine-tuning, and no extra dataset required.
- **High Speedup and High Fidelity:** Achieves up to **7x** wall-clock acceleration while preserving semantic structure and high-frequency details.
- **Cross-Modality Generalizability:** Designed to extend from 2D image generation to 3D video generation.

## News
**2026-03-25:** The public code release now includes inference support for **FLUX.1-dev** and **FLUX.2-klein-base-9B**. 🚀🚀

**2026-03-11:** Our paper is now available on [**arXiv**](https://arxiv.org/abs/2603.10744), and the [**Project Page**](https://wenhao-sun77.github.io/JiT/) is live. 🎉🎉

**2026-02-21:** **Just-in-Time (JiT)** has been accepted to **CVPR 2026**. 🥳🥳

**Coming Soon:** Additional backbones are planned for future release, including **Qwen-image** and **HunyuanVideo 1.5**.

## Getting Started
Create a dedicated conda environment, install the required packages, and enter the repository:

```bash
git clone https://github.com/Wenhao-Sun77/Just-in-Time.git
cd Just-in-Time
conda create -n jit python=3.10 -y
conda activate jit
pip install -r requirement.txt
```

## Repository Structure
```text
.
├── flux/
│   ├── infer.py
│   └── pipeline_flux_JiT.py
├── flux2-klein-base-9B/
│   ├── infer_flux2.py
│   └── pipeline_flux2_klein_JiT.py
├── requirement.txt
└── README.md
```

## Quick Start
Before running the examples, please update `model_path` in the corresponding inference script to the local checkpoint path of your pretrained model.
You can select the CUDA device from the command line through `--gpu_id`.

### FLUX.1-dev
Run the FLUX.1-dev demo with a predefined JiT preset:

```bash
python flux/infer.py --preset default_4x --gpu_id 0
python flux/infer.py --preset default_7x --gpu_id 0
```

The script will:
- load `FluxPipeline_JiT`
- generate a `1024x1024` image from the built-in prompt
- save the result to `./outputs/`

### FLUX.2-klein-base-9B
Run the FLUX.2-klein-base-9B demo with a predefined JiT preset:

```bash
python flux2-klein-base-9B/infer_flux2.py --preset default_4x --gpu_id 0
python flux2-klein-base-9B/infer_flux2.py --preset default_7x --gpu_id 0
```

The script will:
- load `Flux2KleinPipeline_JiT`
- generate a `1024x1024` image from the built-in prompt
- save the result to `./outputs_flux2/`

## JiT Presets
Both FLUX pipelines currently expose two ready-to-use presets through `pipeline.set_params(...)`:

| Preset | Total Steps | Stage Ratios | Sparsity Ratios | Notes |
| --- | ---: | --- | --- | --- |
| `default_4x` | `18` | `[0.4, 0.65, 1.0]` | `[0.35, 0.62, 1.0]` | More conservative acceleration |
| `default_7x` | `11` | `[0.4, 0.65, 1.0]` | `[0.32, 0.6, 1.0]` | More aggressive acceleration |

Actual speedup depends on the GPU, resolution, prompt, runtime backend, and checkpoint implementation.


## Custom Configuration
In addition to presets, you can provide custom JiT parameters through `pipeline.set_params(...)`.

Example for `FLUX.1-dev`:

```python
pipeline.set_params(
    total_steps=11,
    stage_ratios=[0.4, 0.65, 1.0],
    sparsity_ratios=[0.32, 0.6, 1.0],
    use_checkerboard_init=True,
    use_adaptive=True,
    use_beta_sigmas=True,
    alpha=1.4,
    beta=0.42,
    microflow_relax_steps=3,
)
```

Example for `FLUX.2-klein-base-9B`:

```python
pipeline.set_params(
    total_steps=11,
    stage_ratios=[0.4, 0.65, 1.0],
    sparsity_ratios=[0.32, 0.6, 1.0],
    use_checkerboard_init=True,
    use_adaptive=True,
    microflow_relax_steps=3,
)
```

## Timestep Schedule

JiT uses a Beta distribution with `alpha=1.4` and `beta=0.42` to allocate more timesteps to the early denoising stages, which helps preserve a stable global structure during generation.

This scheduling strategy is not necessarily optimal. In practice, it improves semantic stability, but may slightly sacrifice fine-grained details. Suggestions and pull requests for improved scheduling strategies are welcome!

For **FLUX.2-klein-base-9B**, we do not apply this custom beta schedule. Its native timestep schedule already satisfies the coarse-to-fine requirement of JiT. 


## Citation
If you find this project useful in your research, please consider citing it:

```bibtex
@misc{sun2026justintimetrainingfreespatialacceleration,
      title={Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers}, 
      author={Wenhao Sun and Ji Li and Zhaoqiang Liu},
      year={2026},
      eprint={2603.10744},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.10744}
}
```

## Acknowledgments

This repository is built upon and utilizes the [Diffusers](https://github.com/huggingface/diffusers) repository.

We would like to express our sincere thanks to the authors and contributors of [Diffusers](https://github.com/huggingface/diffusers) for their incredible work, which greatly enhanced the development of this repository.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.