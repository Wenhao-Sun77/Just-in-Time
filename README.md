# Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.10744-b31b1b.svg)](https://arxiv.org/abs/2603.10744)
[![Conference](https://img.shields.io/badge/Accepted-CVPR_2026-success)](https://cvpr.thecvf.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[🔥 Project Page (Coming Soon)](#)** | **[📄 Paper](https://arxiv.org/abs/2603.10744)** 

> **Notice:** JiT has been accepted by CVPR2026, and we are actively cleaning up the codebase for public release. The full code will be available soon. Please ⭐ **Star** this repository to stay updated!

## 📖 Introduction
**Just-in-Time (JiT)** is a novel, model-agnostic framework designed for the highly efficient inference of Diffusion Transformers (DiTs). By leveraging a Spatially Approximated Generative ODE (SAG-ODE) and a Deterministic Micro-Flow (DMF), JiT dynamically allocates computational resources to high-information-density spatial regions without requiring any architectural modifications or model fine-tuning.

### ✨ Key Features
- **Training-Free:** Plug-and-play acceleration for pre-trained DiTs. No distillation, no fine-tuning, no dataset required.
- **High Speedup & High Fidelity:** Achieves up to **7×** actual wall-clock speedup while strictly preserving complex semantic structures and high-frequency details.
- **Cross-Modality Generalizability:** Seamlessly scales from 2D image generation to 3D video generation.

## 🚀 Supported Backbones & Showcases
Our method is highly extensible. The upcoming release will include out-of-the-box support for the following state-of-the-art models:
- [x] **FLUX.1-dev**、**FLUX.2-klein-base-9B** (Text-to-Image)
- [x] **Qwen-image** (Text-to-Image)
- [x] **HunyuanVideo 1.5** (Text-to-Video)
