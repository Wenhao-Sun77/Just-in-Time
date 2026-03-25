"""
Just-in-Time (JiT): Training-Free Spatial Acceleration for Diffusion Transformers
CVPR 2026

Pipeline implementation for FLUX2-Klein model with Spatially Approximated Generative ODE (SAG-ODE)
and Deterministic Micro-Flow (DMF) stage transitions.

Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, retrieve_timesteps, compute_empirical_mu, retrieve_latents


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


class Flux2KleinPipeline_JiT(Flux2KleinPipeline):

    def set_params(
        self,
        preset=None,
        total_steps=None,
        sparsity_ratios=None,
        stage_ratios=[0.4, 0.65, 1.0],
        use_adaptive=True,
        use_checkerboard_init=True,
        microflow_relax_steps=3,
    ):
        """
        Configure JiT pipeline parameters for FLUX2-Klein.

        Args:
            preset: Preset configuration ('default_4x', 'default_7x', or None for custom)
            total_steps: Total ODE integration steps
            stage_ratios: Stage transition boundaries
            sparsity_ratios: Token density per stage
            use_adaptive: Enable variance-driven token activation
            use_checkerboard_init: Use checkerboard pattern for initial anchors
            microflow_relax_steps: DMF interpolation substeps
        """
        if preset == 'default_7x':
            total_steps = 11
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.32, 0.6, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            microflow_relax_steps = 3
        elif preset == 'default_4x':
            total_steps = 18
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.35, 0.62, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            microflow_relax_steps = 3

        if total_steps is None:
            raise ValueError("total_steps must be specified or use a preset")

        self.num_stages = len(stage_ratios)
        self.params = {
            "total_steps": total_steps,
            "stage_ratios": stage_ratios,
            "sparsity_ratios": sparsity_ratios,
            "stage_steps": [int(total_steps * r) for r in stage_ratios],
            "use_adaptive": use_adaptive,
            "use_checkerboard_init": use_checkerboard_init,
            "microflow_relax_steps": microflow_relax_steps,
        }

        self.active_token_indices = {}

        # Precompute coordinate grid for interpolation
        self._coords_full = None
        self._H_packed = None
        self._W_packed = None

        print(f"JiT Configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Stage divisions: {self.params['stage_steps']}")
        print(f"  Token densities: {sparsity_ratios}")
        print(f"  Adaptive densification: {use_adaptive}")

    def _ratio_of_stage(self, stage_k: int) -> float:
        ratios = self.params["sparsity_ratios"]
        if ratios[0] <= ratios[-1]:  # Ascending: [few, ..., many]
            return ratios[self.num_stages - 1 - stage_k]
        else:  # Descending: [many, ..., few]
            return ratios[stage_k]

    def _create_sparse_grid(
        self,
        H_packed: int,
        W_packed: int,
        sparsity_ratio: float,
        device: torch.device,
        use_checkerboard: bool = False
    ) -> torch.Tensor:
        """
        Initialize sparse anchor token set (Ω_K) for SAG-ODE.

        Args:
            H_packed, W_packed: Token grid dimensions
            sparsity_ratio: Target token density [0, 1]
            device: torch device
            use_checkerboard: Use fixed checkerboard pattern

        Returns:
            1D tensor of anchor token indices
        """
        N_packed = H_packed * W_packed
        m_k = int(N_packed * sparsity_ratio)

        if use_checkerboard:
            i_coords = torch.arange(H_packed, device=device)
            j_coords = torch.arange(W_packed, device=device)
            ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')

            all_indices = torch.arange(N_packed, device=device)
            mask_core = (ii % 2 == 0) & (jj % 2 == 0)
            mask_boundary = (ii == 0) | (ii == H_packed - 1) | (jj == 0) | (jj == W_packed - 1)
            mask_combined = mask_core | mask_boundary
            indices = all_indices[mask_combined.flatten()]
        else:
            stride = max(1, int(np.sqrt(1.0 / sparsity_ratio)))
            grid_h = torch.arange(0, H_packed, stride, device=device)
            grid_w = torch.arange(0, W_packed, stride, device=device)

            if len(grid_h) == 0 or grid_h[-1] != H_packed - 1:
                grid_h = torch.cat([grid_h, torch.tensor([H_packed-1], device=device)])
            if len(grid_w) == 0 or grid_w[-1] != W_packed - 1:
                grid_w = torch.cat([grid_w, torch.tensor([W_packed-1], device=device)])

            mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
            indices = mesh_h.flatten() * W_packed + mesh_w.flatten()

        if len(indices) < m_k:
            all_indices_set = set(range(N_packed))
            available = list(all_indices_set - set(indices.tolist()))
            if available:
                n_supplement = min(m_k - len(indices), len(available))
                supplement = torch.tensor(
                    np.random.choice(available, n_supplement, replace=False),
                    device=device
                )
                indices = torch.cat([indices, supplement])
        elif len(indices) > m_k:
            perm = torch.randperm(len(indices), device=device)[:m_k]
            indices = indices[perm]

        return indices.long()
    
    def _calculate_blur_params(self, sparsity_ratio, c=0.4):
        if sparsity_ratio <= 0.0:
            return 3, 1.0
        if sparsity_ratio >= 1.0:
            return 3, 1.0

        characteristic_distance_L = 1.0 / np.sqrt(sparsity_ratio)
        sigma = c * characteristic_distance_L
        sigma = np.clip(sigma, a_min=1.0, a_max=10.0) 
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        
        return kernel_size, sigma

    def _precompute_coords(self, H_packed, W_packed, device):
        """Precompute coordinate grid for interpolation."""
        if self._coords_full is None or self._H_packed != H_packed or self._W_packed != W_packed:
            coords_y, coords_x = torch.meshgrid(
                torch.arange(H_packed, device=device),
                torch.arange(W_packed, device=device),
                indexing="ij"
            )
            self._coords_full = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=-1)
            self._H_packed = H_packed
            self._W_packed = W_packed

    def _irregular_interpolation(self, y_active, active_indices, N_full, d, H_packed, W_packed, device, dtype):
        """
        Spatial interpolation operator for SAG-ODE velocity approximation.

        Args:
            y_active: [B, M, d] anchor token values
            active_indices: [M] anchor positions
            N_full: Total token count
            d: Token dimension
            H_packed, W_packed: Grid dimensions
            device, dtype: torch device and dtype

        Returns:
            [B, N_full, d] interpolated full-dimensional tensor
        """
        if active_indices.numel() == 0:
            return torch.zeros(y_active.size(0), N_full, d, device=device, dtype=dtype)

        B, M, _ = y_active.shape

        # Step 1: Nearest neighbor fill using precomputed coordinates
        coords_active = self._coords_full[active_indices]
        dist = torch.cdist(self._coords_full.float(), coords_active.float(), p=2)
        nearest_idx = dist.argmin(dim=-1)
        y_active_expanded = y_active.permute(1, 0, 2)
        gathered = y_active_expanded[nearest_idx]
        y_full_nearest = gathered.permute(1, 0, 2).contiguous()

        # Step 2: Masked Gaussian blur
        y_full_2d_nearest = y_full_nearest.reshape(B, H_packed, W_packed, d).permute(0, 3, 1, 2)
        sparsity_ratio = len(active_indices) / N_full
        kernel_size, sigma = self._calculate_blur_params(sparsity_ratio, c=0.4)
        if kernel_size % 2 == 0:
            kernel_size += 1

        y_full_2d_blur = gaussian_blur(
            y_full_2d_nearest,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
        )

        active_mask_1d = torch.zeros(N_full, device=device, dtype=dtype)
        active_mask_1d[active_indices] = 1.0
        active_mask_2d = active_mask_1d.reshape(1, 1, H_packed, W_packed).expand(B, -1, -1, -1)
        inactive_mask_2d = 1.0 - active_mask_2d

        y_full_2d_final = y_full_2d_nearest * active_mask_2d + y_full_2d_blur * inactive_mask_2d
        y_full = y_full_2d_final.permute(0, 2, 3, 1).reshape(B, N_full, d)

        return y_full  
 
    def _compute_importance_map(self, y_full, velocity_full, active_indices, H_packed, W_packed):
        """
        Zero-cost importance map for adaptive token activation.

        Computes local variance of cached velocity to identify high-frequency regions.

        Args:
            y_full: [B, N, d] full latent state
            velocity_full: [B, N, d] cached velocity field
            active_indices: Current anchor indices
            H_packed, W_packed: Grid dimensions

        Returns:
            [H_packed, W_packed] variance-based importance map
        """
        B, N, d = y_full.shape
        device = y_full.device

        v_2d = velocity_full.reshape(B, H_packed, W_packed, d).permute(0, 3, 1, 2)

        kernel_size = 3
        mean = F.avg_pool2d(v_2d, kernel_size, stride=1, padding=kernel_size//2)
        var = F.avg_pool2d(v_2d**2, kernel_size, stride=1, padding=kernel_size//2) - mean**2
        importance = var.mean(dim=1).squeeze(0)

        return importance
    
    def _adaptive_densify(self, current_indices, target_count, importance_map, H_packed, W_packed):
        """
        Variance-driven token activation for stage transitions.

        Args:
            current_indices: Current anchor indices
            target_count: Target active token count
            importance_map: [H, W] importance map
            H_packed, W_packed: Grid dimensions

        Returns:
            Updated active token indices
        """
        device = importance_map.device
        N_packed = H_packed * W_packed

        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + 1e-8
        )

        importance_flat = importance_map.flatten()
        mask = torch.ones(N_packed, dtype=torch.bool, device=device)
        mask[current_indices] = False

        candidate_indices = torch.arange(N_packed, device=device)[mask]
        candidate_importance = importance_flat[candidate_indices]

        num_to_add = target_count - len(current_indices)

        if num_to_add <= 0:
            return current_indices

        if num_to_add >= len(candidate_indices):
            new_indices = torch.cat([current_indices, candidate_indices])
        else:
            probabilities = candidate_importance / (candidate_importance.sum() + 1e-8)
            _, top_k_indices = torch.topk(probabilities, num_to_add)
            selected = candidate_indices[top_k_indices]
            new_indices = torch.cat([current_indices, selected])

        return new_indices.long()

    def _extract_active_tokens(self, y_full, active_indices):
        """Extract active tokens from full state."""
        return y_full[:, active_indices, :], active_indices

    def _compute_variance_schedule(self, timestep):
        """Variance schedule mapping."""
        return (timestep * 1e-3) ** 2

    # ========== Deterministic Micro-Flow (DMF) ==========

    def _microflow_bridge(self, y_full, new_indices, y_target_new):
        """
        Deterministic Micro-Flow (DMF) for smooth stage transitions.

        Args:
            y_full: [B, N, d] full latent state
            new_indices: Newly activated token indices
            y_target_new: [B, M_new, d] target values for new tokens

        Returns:
            Updated full latent state
        """
        if new_indices.numel() == 0:
            return y_full

        steps = self.params.get("microflow_relax_steps", 3)
        if steps <= 0:
            y_full[:, new_indices, :] = y_target_new
        else:
            current = y_full[:, new_indices, :].clone()
            weight = 1.0 / steps
            y_full[:, new_indices, :] = (1 - weight) * current + weight * y_target_new

        return y_full

    # ========== Auxiliary Methods ==========

    def _prepare_latent_image_ids(self, indices, H_packed, W_packed, device, dtype):
        """
        Generate positional encodings for active tokens (FLUX2 uses 4D coords).

        Args:
            indices: Active token indices
            H_packed, W_packed: Grid dimensions
            device, dtype: torch device and dtype

        Returns:
            [M, 4] positional IDs (t, h, w, l)
        """
        t = torch.zeros(H_packed, W_packed, 1, device=device, dtype=dtype)
        h = torch.arange(H_packed, device=device, dtype=dtype).view(H_packed, 1, 1).expand(-1, W_packed, -1)
        w = torch.arange(W_packed, device=device, dtype=dtype).view(1, W_packed, 1).expand(H_packed, -1, -1)
        l = torch.zeros(H_packed, W_packed, 1, device=device, dtype=dtype)

        latent_image_ids = torch.cat([t, h, w, l], dim=-1)
        latent_image_ids = latent_image_ids.reshape(H_packed * W_packed, 4)
        return latent_image_ids[indices]

    def _predict_x0_latent(self, latents, noise_pred, timestep):
        """
        Tweedie prediction for clean data estimation.

        Args:
            latents: Current state x_t
            noise_pred: Velocity field v_t
            timestep: Current time t

        Returns:
            Predicted clean latent x_0
        """
        latents_dtype = latents.dtype
        latents_x0 = (
            latents.to(torch.float32) -
            noise_pred.to(torch.float32) * timestep.to(torch.float32) * 1e-3
        ).to(latents_dtype)
        return latents_x0

    @torch.no_grad()
    #@replace_example_docstring(Flux2KleinPipeline.EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        # 1. Inputs & JiT Params
        # Must be initialized via set_params before call, but we check/set default here if needed
        if not hasattr(self, "params"):
             self.set_params(total_steps=num_inference_steps)

        total_steps = self.params["total_steps"]
        stage_steps = self.params["stage_steps"]
        use_adaptive = self.params["use_adaptive"]
        use_checkerboard_init = self.params["use_checkerboard_init"]

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        
        # We need full latents first
        
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        
        # At this point, latents is [B, N_packed, C], latent_ids is [B, N_packed, 4]
        # We define y_full as this initial latent state
        y_full = latents.clone()
        B, N_packed, D = y_full.shape
        # Flux2Klein prepare latents: height = 2 * (int(height) // (self.vae_scale_factor * 2))
        # Packed latents shape [B, H//2 * W//2, C]
        # So H_packed (grid dim) is height // (self.vae_scale_factor * 2)
        H_packed = int(height) // (self.vae_scale_factor * 2)
        W_packed = int(width) // (self.vae_scale_factor * 2)
        
        # Verify N_packed
        if N_packed != H_packed * W_packed:
             # Just in case there's some unexpected padding/sizing, we warn or check
             logger.warning(f"Warning: Calculated N_packed {H_packed*W_packed} != actual {N_packed}")

        # Prepare condition images (these are kept dense/untouched in JiT logic usually)
        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # Pre-generate global noise for consistent randomness
        global_noise = torch.randn(
            y_full.shape[0], y_full.shape[1], y_full.shape[2],
            device=device, dtype=y_full.dtype
        )

        # Initialize sparse anchor token set (Ω_K)
        current_stage = self.num_stages - 1
        ratio0 = self._ratio_of_stage(current_stage)
        current_indices = self._create_sparse_grid(H_packed, W_packed, ratio0, device, use_checkerboard_init)
        self.active_token_indices[current_stage] = current_indices
        activation_ratio = len(current_indices) / N_packed
        print(f"Stage {current_stage}: Initialized {len(current_indices)} tokens (density={activation_ratio:.2%})")

        # Precompute coordinate grid for interpolation optimization
        self._precompute_coords(H_packed, W_packed, device)

        # Initialize velocity cache for zero-order hold approximation
        last_velocity_full = None

        # Main SAG-ODE denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Check for stage transition
                t_curr = timesteps[i]

                target_stage = 0
                for s_idx, s_step in enumerate(stage_steps):
                    if i < s_step:
                        target_stage = self.num_stages - 1 - s_idx
                        break

                if target_stage < current_stage:
                    print(f"\nStep {i}: Stage transition {current_stage} → {target_stage}")
                    target_count = int(N_packed * self._ratio_of_stage(target_stage))

                    if use_adaptive and i > 0 and last_velocity_full is not None:
                        print(f"  Using adaptive densification (variance-based)")
                        importance_map = self._compute_importance_map(
                            y_full, last_velocity_full, current_indices,
                            H_packed, W_packed
                        )
                        new_indices = self._adaptive_densify(
                            current_indices, target_count, importance_map,
                            H_packed, W_packed
                        )
                    else:
                        print(f"  Using fixed grid expansion")
                        new_indices = self._create_sparse_grid(
                            H_packed, W_packed, self._ratio_of_stage(target_stage), device, use_checkerboard_init
                        )

                    newly_activated = new_indices[~torch.isin(new_indices, current_indices)]

                    # DMF: Compute target state for newly activated tokens
                    B_sz, d_sz = y_full.shape[0], y_full.shape[-1]
                    
                    sigma_target = torch.sqrt(self._compute_variance_schedule(t_curr))
                    noise = global_noise[:, newly_activated, :] * sigma_target
                    
                    t_before = timesteps[i-1]
                    latents_x0 = self._predict_x0_latent(last_y, last_velocity_full, t_before)
                    x0_interpolated = self._irregular_interpolation(
                        latents_x0[:, current_indices, :], current_indices, N_packed, d_sz, H_packed, W_packed, device, y_full.dtype
                    )
                    y_target_new = x0_interpolated[:, newly_activated, :]*(1-sigma_target) + noise

                    y_full = self._microflow_bridge(y_full, newly_activated, y_target_new)

                    current_indices = new_indices
                    self.active_token_indices[target_stage] = current_indices
                    current_stage = target_stage
                    print(f"  Newly activated tokens: {len(newly_activated)}")

                # Extract active tokens and compute velocity
                y_active, _ = self._extract_active_tokens(y_full, current_indices)
                ids_active = self._prepare_latent_image_ids(current_indices, H_packed, W_packed, device, y_full.dtype)
                
                # Expand IDs to batch
                ids_active = ids_active.unsqueeze(0).expand(B, -1, -1)

                latent_model_input = y_active.to(self.transformer.dtype)
                latent_image_ids = ids_active

                if image_latents is not None:
                    latent_model_input = torch.cat([y_active, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([ids_active, image_latent_ids], dim=1)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                # Separate output (only keep generation part)
                noise_pred_active = noise_pred[:, : y_active.size(1) :]

                # Irregular Interpolation to Full Grid
                if current_stage > 0:
                    velocity_full_step = self._irregular_interpolation(
                        noise_pred_active, current_indices, N_packed, y_full.shape[-1],
                        H_packed, W_packed, device, y_full.dtype
                    )
                else:
                    velocity_full_step = torch.zeros_like(y_full)
                    velocity_full_step[:, current_indices, :] = noise_pred_active

                last_velocity_full = velocity_full_step.clone()
                last_y = y_full

                # Handle CFG (Note: JiT in pipeline_flux_JiT apparently handles CFG inside transformer block or separately, but here we used irregular interpolation on the result. If CFG is needed, it should be done on active tokens before interpolation)
                # Flux2Klein applies CFG on `noise_pred`.
                if self.do_classifier_free_guidance:
                     # This requires running uncond pass.
                     # For JiT, efficient way is to run uncond only on active tokens too.
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    neg_noise_pred_active = neg_noise_pred[:, : y_active.size(1) :]
                    noise_pred_active = neg_noise_pred_active + guidance_scale * (noise_pred_active - neg_noise_pred_active)
                    
                    # Re-interpolate proper velocity after CFG
                    if current_stage > 0:
                        velocity_full_step = self._irregular_interpolation(
                            noise_pred_active, current_indices, N_packed, y_full.shape[-1],
                            H_packed, W_packed, device, y_full.dtype
                        )
                    else:
                        velocity_full_step[:, current_indices, :] = noise_pred_active

                    last_velocity_full = velocity_full_step

                # Scheduler Step
                # Flux2Klein uses `self.scheduler.step(noise_pred, ...)`
                # We need to step the FULL latent `y_full` using `velocity_full_step`
                
                latents_dtype = y_full.dtype
                y_full = self.scheduler.step(velocity_full_step, t, y_full, return_dict=False)[0]

                if y_full.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        y_full = y_full.to(latents_dtype)
                

                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                             callback_kwargs[k] = y_full
                        else:
                             callback_kwargs[k] = locals().get(k)
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    y_full = callback_outputs.pop("latents", y_full)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.set_postfix(
                    stage=current_stage,
                    active=len(current_indices),
                    total=N_packed,
                    ratio=f"{len(current_indices)/N_packed:.1%}"
                )
                progress_bar.update()

        self._current_timestep = None

        # Final Unpack
        # Use y_full (fully dense)
        # We need latent_ids for full grid. Flux2Klein `_unpack_latents_with_ids` expects usage of `latent_ids`
        # latent_ids was created at start and is full.
        
        latents = self._unpack_latents_with_ids(y_full, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
