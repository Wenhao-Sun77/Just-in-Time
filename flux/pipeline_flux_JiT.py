"""
Just-in-Time (JiT): Training-Free Spatial Acceleration for Diffusion Transformers
CVPR 2026

Pipeline implementation for FLUX model with Spatially Approximated Generative ODE (SAG-ODE)
and Deterministic Micro-Flow (DMF) stage transitions.
"""

import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision.transforms.functional import gaussian_blur
import inspect

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class FluxPipeline_JiT(FluxPipeline):

    def set_params(
        self,
        preset=None,
        total_steps=None,
        stage_ratios=[0.4, 0.65, 1.0],
        sparsity_ratios=None,
        use_checkerboard_init=True,
        use_adaptive=True,
        use_beta_sigmas=True,
        alpha=1.4,
        beta=0.42,
        microflow_relax_steps=3,
    ):
        """
        Configure JiT pipeline parameters.

        Args:
            preset: Preset configuration ('default_4x', 'default_7x', or None for custom)
            total_steps: Total ODE integration steps
            stage_ratios: Stage transition boundaries (e.g., [0.3, 0.6, 1.0])
            sparsity_ratios: Token density per stage (e.g., [0.1, 0.4, 1.0])
            use_adaptive: Enable variance-driven token activation
            use_checkerboard_init: Use checkerboard pattern for initial anchors
            microflow_relax_steps: DMF interpolation substeps
            use_beta_sigmas: Use Beta distribution for timestep schedule
        """
        # Apply preset configurations
        if preset == 'default_7x':
            total_steps = 11
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.32, 0.6, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            use_beta_sigmas = True
            alpha = 1.4
            beta = 0.425
            microflow_relax_steps = 3
        elif preset == 'default_4x':
            total_steps = 18
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.35, 0.62, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            use_beta_sigmas = True
            alpha = 1.4
            beta = 0.425
            microflow_relax_steps = 3

        # Validate required parameters
        if total_steps is None:
            raise ValueError("total_steps must be specified or use a preset")
        if sparsity_ratios is None:
            sparsity_ratios = [0.35, 0.62, 1.0]
        self.num_stages = len(stage_ratios)
        self.params = {
            "total_steps": total_steps,
            "stage_ratios": stage_ratios,
            "sparsity_ratios": sparsity_ratios,
            "stage_steps": [int(total_steps * r) for r in stage_ratios],
            "use_adaptive": use_adaptive,
            "use_checkerboard_init": use_checkerboard_init,
            "microflow_relax_steps": microflow_relax_steps,
            "use_beta_sigmas": use_beta_sigmas,
            "alpha": alpha,
            "beta": beta,
        }

        self.active_token_indices = {}

        # Precompute coordinate grid for interpolation (will be set during inference)
        self._coords_full = None
        self._H_packed = None
        self._W_packed = None

        print(f"JiT Configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Stage divisions: {self.params['stage_steps']}")
        print(f"  Token densities: {sparsity_ratios}")
        print(f"  Adaptive densification: {use_adaptive}")

    # ========== Pack/Unpack Operations ==========

    def _pack_latents(self, latents):
        """
        Pack VAE latents into DiT token format.

        Merges 2x2 spatial blocks into channel dimension.

        Args:
            latents: [B, C, H, W] VAE latent (H, W divisible by 2)

        Returns:
            [B, H//2 * W//2, C*4] packed token sequence
        """
        B, C, H, W = latents.shape
        assert H % 2 == 0 and W % 2 == 0, f"Latent size must be divisible by 2, got {H}×{W}"
        
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, H // 2, W // 2, C * 4)
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents

    def _unpack_latents(self, latents, H_packed, W_packed):
        """
        Unpack DiT tokens back to VAE latent format.

        Args:
            latents: [B, H_packed * W_packed, C*4]
            H_packed, W_packed: Packed grid dimensions

        Returns:
            [B, C, H_packed*2, W_packed*2] VAE latent
        """
        B, N, C_packed = latents.shape
        C = C_packed // 4

        latents = latents.reshape(B, H_packed, W_packed, C * 4)
        latents = latents.reshape(B, H_packed, W_packed, C, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C, H_packed * 2, W_packed * 2)
        return latents
    
    def _ratio_of_stage(self, stage_k: int) -> float:
        ratios = self.params["sparsity_ratios"]
        # Allow user to specify ratios in ascending (few -> many) or descending (many -> few) order:
        if ratios[0] <= ratios[-1]:  # Ascending: [few, ..., many]
            # Stage 2 (early/sparse) -> ratios[0]; Stage 0 (late/dense) -> ratios[-1]
            return ratios[self.num_stages - 1 - stage_k]
        else:  # Descending: [many, ..., few]
            return ratios[stage_k]

    # ========== Sparse Anchor Token Initialization ==========

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

        Two strategies:
        - Checkerboard: Fixed stride-2 grid + boundary (recommended for ratio ≈ 0.25)
        - Adaptive: Dynamic stride + random sampling (flexible for any ratio)

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
            # Checkerboard: stride-2 grid + boundary coverage
            i_coords = torch.arange(H_packed, device=device)
            j_coords = torch.arange(W_packed, device=device)
            ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')

            all_indices = torch.arange(N_packed, device=device)
            mask_core = (ii % 2 == 0) & (jj % 2 == 0)
            mask_boundary = (ii == 0) | (ii == H_packed - 1) | (jj == 0) | (jj == W_packed - 1)
            mask_combined = mask_core | mask_boundary
            indices = all_indices[mask_combined.flatten()]
        else:
            # Adaptive: dynamic stride based on sparsity
            stride = max(1, int(np.sqrt(1.0 / sparsity_ratio)))
            grid_h = torch.arange(0, H_packed, stride, device=device)
            grid_w = torch.arange(0, W_packed, stride, device=device)

            if len(grid_h) == 0 or grid_h[-1] != H_packed - 1:
                grid_h = torch.cat([grid_h, torch.tensor([H_packed-1], device=device)])
            if len(grid_w) == 0 or grid_w[-1] != W_packed - 1:
                grid_w = torch.cat([grid_w, torch.tensor([W_packed-1], device=device)])

            mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
            indices = mesh_h.flatten() * W_packed + mesh_w.flatten()

        # Adjust to exact target count
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
        """
        Compute Gaussian blur parameters based on anchor token density.

        Args:
            sparsity_ratio: Active token density
            c: Hyperparameter controlling blur intensity

        Returns:
            (kernel_size, sigma) tuple
        """
        if sparsity_ratio <= 0.0 or sparsity_ratio >= 1.0:
            return 3, 1.0

        characteristic_distance_L = 1.0 / np.sqrt(sparsity_ratio)
        sigma = c * characteristic_distance_L
        sigma = np.clip(sigma, a_min=1.0, a_max=10.0)
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1

        return kernel_size, sigma

    def _precompute_coords(self, H_packed, W_packed, device):
        """Precompute coordinate grid for interpolation to avoid repeated computation."""
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

        Implements zero-order hold with masked Gaussian smoothing:
        1. Nearest neighbor fill from anchor tokens
        2. Gaussian blur on inactive regions
        3. Preserve exact values at anchor positions

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

        # Create mask: preserve anchor values, use blur for inactive regions
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

        Selects top-K inactive tokens based on importance to expand active set.

        Args:
            current_indices: Current anchor indices
            target_count: Target active token count
            importance_map: [H, W] variance-based importance
            H_packed, W_packed: Grid dimensions

        Returns:
            Updated active token indices
        """
        device = importance_map.device
        N_packed = H_packed * W_packed
        
        # Normalize importance
        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + 1e-8
        )
        
        # Select candidates
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
        """Extract active tokens"""
        return y_full[:, active_indices, :], active_indices

    # ========== Deterministic Micro-Flow (DMF) ==========

    def _microflow_bridge(self, y_full, new_indices, y_target_new):
        """
        Deterministic Micro-Flow (DMF) for smooth stage transitions.

        Linearly interpolates newly activated tokens from current to target state.

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
        Generate positional encodings for active tokens.

        Args:
            indices: Active token indices
            H_packed, W_packed: Grid dimensions
            device, dtype: torch device and dtype

        Returns:
            [M, 3] positional IDs (batch_idx, height_pos, width_pos)
        """
        latent_image_ids = torch.zeros(H_packed, W_packed, 3, device=device, dtype=dtype)
        latent_image_ids[..., 0] = 0
        latent_image_ids[..., 1] = torch.arange(H_packed, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(W_packed, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(H_packed * W_packed, 3)
        return latent_image_ids[indices]

    def _predict_x0_latent(self, latents, noise_pred, timestep):
        """
        Tweedie prediction for clean data estimation.

        Predicts x_0 from current state: x_0 = x_t - t * v_t

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
    
    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None, desc=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        config = self._progress_bar_config.copy()
        if desc is not None:
            config["desc"] = desc
        if iterable is not None:
            return tqdm(iterable, **config)
        elif total is not None:
            return tqdm(total=total, **config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs
    ):
        """
        JiT inference pipeline with SAG-ODE and DMF stage transitions.

        Args:
            prompt: Text prompt(s)
            height, width: Output image dimensions
            num_inference_steps: Total ODE integration steps
            guidance_scale: Classifier-free guidance scale
            Other args: Standard diffusers pipeline parameters

        Returns:
            FluxPipelineOutput with generated images
        """
        self.enable_vae_tiling()
        self.vae.to(self._execution_device)

        # Extract parameters
        total_steps = self.params["total_steps"]
        stage_steps = self.params["stage_steps"]
        sparsity_ratios = self.params["sparsity_ratios"]
        use_adaptive = self.params["use_adaptive"]
        use_beta_sigmas = self.params["use_beta_sigmas"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        # Reset storage
        self.active_token_indices = {}
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Height and width must be divisible by 16, got {height}×{width}")
        
        # VAE latent dimensions
        H_latent = height // self.vae_scale_factor
        W_latent = width // self.vae_scale_factor
        
        # Packed token dimensions
        H_packed = H_latent // 2
        W_packed = W_latent // 2
        N_packed = H_packed * W_packed
        
        print(f"\n{'='*70}")
        print(f"JiT: Training-Free Spatial Acceleration for DiTs")
        print(f"Image resolution: {height}×{width}")
        print(f"VAE latent: {H_latent}×{W_latent}")
        print(f"Packed tokens: {H_packed}×{W_packed} ({N_packed} tokens)")
        print(f"Token densities: {sparsity_ratios}")
        print(f"Adaptive densification: {use_adaptive}")
        print(f"{'='*70}\n")

        # Check inputs
        self.check_inputs(
            prompt, 
            None, 
            height, 
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False

        # Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Encode prompts
        lora_scale = self.joint_attention_kwargs.get("scale", None)
        (prompt_embeds, 
         pooled_prompt_embeds, 
         text_ids
         ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Initialize latents
        num_channels_latents = self.transformer.config.in_channels // 4

        if latents is None:
            # Initialize noise in VAE latent space
            latents_shape = (
                batch_size * num_images_per_prompt,
                num_channels_latents,
                H_latent,
                W_latent
            )
            latents_2d = randn_tensor(
                latents_shape,
                generator=generator,
                device=device,
                dtype=prompt_embeds.dtype
            )
        else:
            latents_2d = latents

        # Pack into token sequence
        y_full = self._pack_latents(latents_2d)

        # Pre-generate global noise for consistent randomness
        global_noise = torch.randn(
            y_full.shape[0],
            y_full.shape[1],
            y_full.shape[2],
            device=device,
            dtype=y_full.dtype,
        )

        # Prepare timesteps
        num_inference_steps = total_steps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        if use_beta_sigmas:
            sigmas_beta = self.scheduler._convert_to_beta(
                in_sigmas=sigmas,
                num_inference_steps=total_steps,
                alpha=alpha,
                beta=beta,
            )
            self.scheduler.sigmas = torch.from_numpy(sigmas_beta).to(dtype=torch.float32, device=device)
            timesteps = (self.scheduler.sigmas * self.scheduler.config.num_train_timesteps)
        else:   
            image_seq_len = y_full.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
            )
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(batch_size * num_images_per_prompt)
        else:
            guidance = None

        # Initialize sparse anchor token set (Ω_K)
        use_checkerboard = self.params["use_checkerboard_init"]
        current_stage = self.num_stages - 1
        ratio0 = self._ratio_of_stage(current_stage)
        current_indices = self._create_sparse_grid(H_packed, W_packed, ratio0, device, use_checkerboard)
        self.active_token_indices[current_stage] = current_indices
        print(f"Stage {current_stage}: Initialized {len(current_indices)} tokens (density={ratio0:.2%})")

        # Precompute coordinate grid for interpolation optimization
        self._precompute_coords(H_packed, W_packed, device)

        # Initialize velocity cache for zero-order hold approximation
        last_velocity_full = None

        # Main SAG-ODE denoising loop
        with self.progress_bar(total=total_steps, desc="Denoising") as progress_bar:
            for i in range(total_steps):
                if self.interrupt:
                    continue
                t_0 = timesteps[0].clone()
                t_0.zero_()
                t_curr = timesteps[i]
                t_next = timesteps[i+1] if i+1 < len(timesteps) else t_0
                dt = (t_next - t_curr) * 1e-3

                # Check for stage transition
                target_stage = 0
                for s_idx, s_step in enumerate(stage_steps):
                    if i < s_step:
                        target_stage = self.num_stages - 1 - s_idx
                        break

                if target_stage < current_stage:
                    print(f"\nStep {i}: Stage transition {current_stage} → {target_stage}")
                    
                    target_count = int(N_packed * self._ratio_of_stage(target_stage))

                    if use_adaptive and i > 0:
                        print(f"  Using adaptive densification (variance-based)")
                        importance_map = self._compute_importance_map(
                            y_full, last_velocity_full, current_indices, H_packed, W_packed
                        )

                        new_indices = self._adaptive_densify(
                            current_indices, target_count, importance_map, H_packed, W_packed
                        )
                    else:
                        print(f"  Using fixed grid expansion")
                        new_indices = self._create_sparse_grid(
                            H_packed, W_packed, self._ratio_of_stage(target_stage), device, use_checkerboard
                        )
                    newly_activated = new_indices[~torch.isin(new_indices, current_indices)]

                    # DMF: Compute target state for newly activated tokens
                    B, d = y_full.shape[0], y_full.shape[-1]
                    sigma_target = t_curr * 1e-3
                    noise = global_noise[:, newly_activated, :] * sigma_target

                    t_before = timesteps[i-1]
                    latents_x0 = self._predict_x0_latent(last_y, last_velocity_full, t_before)
                    x0_interpolated = self._irregular_interpolation(
                        latents_x0[:, current_indices, :],
                        current_indices,
                        N_packed,
                        d,
                        H_packed,
                        W_packed,
                        device,
                        y_full.dtype,
                    )
                    y_target_new = x0_interpolated[:, newly_activated, :]*(1-sigma_target) + noise

                    y_full = self._microflow_bridge(y_full, newly_activated, y_target_new)

                    current_indices = new_indices
                    self.active_token_indices[target_stage] = current_indices
                    current_stage = target_stage
                    print(f"  Newly activated tokens: {len(newly_activated)}")

                # Extract active tokens and compute velocity
                y_active, _ = self._extract_active_tokens(y_full, current_indices)
                m_k = y_active.shape[1]

                # DiT forward pass on anchor tokens
                timestep_expanded = t_curr.expand(y_active.shape[0]).to(y_active.dtype)
                latent_ids_active = self._prepare_latent_image_ids(
                    current_indices, H_packed, W_packed, device, y_full.dtype
                )
                
                noise_pred_active = self.transformer(
                    hidden_states=y_active,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids_active,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Spatial interpolation for full velocity field
                if current_stage > 0:
                    velocity_full = self._irregular_interpolation(
                        noise_pred_active, current_indices, N_packed, y_full.shape[-1],
                        H_packed, W_packed, device, y_full.dtype
                    )
                else:
                    velocity_full = torch.zeros_like(y_full)
                    velocity_full[:, current_indices, :] = noise_pred_active

                # Cache
                last_velocity_full = velocity_full
                last_y = y_full.clone()

                # Euler integration step
                y_full = y_full + velocity_full * dt
                
                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {"latents": y_full}
                    callback_outputs = callback_on_step_end(self, i, t_curr, callback_kwargs)
                    y_full = callback_outputs.get("latents", y_full)
                
                progress_bar.update()
                progress_bar.set_postfix({
                    "stage": current_stage,
                    "active": m_k,
                    "total": N_packed,
                    "ratio": f"{m_k/N_packed:.1%}"
                })

        # Decode final output
        if output_type == "latent":
            image = y_full
        else:
            latents_final = self._unpack_latents(y_full, H_packed, W_packed)
            latents_final = (latents_final / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents_final, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        print(f"\n{'='*70}")
        print(f"Generation complete")
        print(f"Final active tokens: {len(current_indices)}/{N_packed} ({len(current_indices)/N_packed:.2%})")
        print(f"{'='*70}\n")

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
