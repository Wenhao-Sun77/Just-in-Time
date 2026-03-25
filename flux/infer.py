from pipeline_flux_JiT import FluxPipeline_JiT
import torch
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX JiT inference")
    parser.add_argument(
        "--preset",
        type=str,
        default="default_4x",
        help="Preset name for pipeline.set_params, e.g. default_4x or default_7x",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device id used for inference",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Device setup
    device = torch.device(f"cuda:{args.gpu_id}")
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)

    # Load pipeline
    model_path = "black-forest-labs/FLUX.1-dev" # Replace this with your local path if you are loading the model offline
    pipeline = FluxPipeline_JiT.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    # Use preset (default_4x or default_7x) or custom parameters
    pipeline.set_params(preset=args.preset)
    
    """       
    # Custom parameters example:
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
     )  """
     
    
    # Configuration
    resolution = 1024
    # Prompt
    prompt = "A grand piano made entirely of transparent, crystal-clear ice, with delicate frost patterns on its surface. It sits in a warm, sunlit concert hall, slowly melting, with water dripping onto the polished wooden floor. Photorealistic, poignant."

    # Generate image
    print(f"Generating: {prompt}")
        
    t0 = time.perf_counter()
    image = pipeline(
        prompt=prompt,
        height=resolution,
        width=resolution,
        max_sequence_length=256,
        generator=generator,
        guidance_scale=3.5
    ).images[0]
    elapsed = time.perf_counter() - t0
    print(f"Cost: {elapsed:.3f}s") 

    # Save output
    output_dir = "./outputs/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prompt[:50]}.png")
    image.save(output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
