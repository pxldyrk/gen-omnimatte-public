"""
Configuration utilities for GenOmnimatte models.

Provides helper functions to create and customize model configurations.
"""

import ml_collections
import torch
from typing import Optional


def get_casper_config(
    model_name: str = "models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP",
    transformer_path: str = "",
    vae_path: str = "",
    lora_path: str = "",
    sample_size: str = "384x672",
    max_video_length: int = 197,
    fps: int = 16,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    denoise_strength: float = 1.0,
    seed: int = 43,
    device: str = "cuda",
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8",
) -> ml_collections.ConfigDict:
    """
    Create a configuration for the Casper (CogVideoX) video inpainting model.

    Args:
        model_name: Path to the pretrained CogVideoX model
        transformer_path: Optional path to custom transformer weights
        vae_path: Optional path to custom VAE weights
        lora_path: Optional path to LoRA weights
        sample_size: Output video size as "HxW" (e.g., "384x672")
        max_video_length: Maximum number of frames to process
        fps: Frames per second for output video
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        denoise_strength: Denoising strength (0.0 to 1.0)
        seed: Random seed for reproducibility
        device: Device to run on ("cuda" or "cpu")
        gpu_memory_mode: Memory optimization mode

    Returns:
        ConfigDict with all model settings
    """
    config = ml_collections.ConfigDict()

    # System config
    config.system = ml_collections.ConfigDict()
    config.system.low_gpu_memory_mode = False
    config.system.weight_dtype = torch.bfloat16
    config.system.seed = seed
    config.system.allow_skipping_error = False
    config.system.device = device
    config.system.gpu_memory_mode = gpu_memory_mode
    config.system.ulysses_degree = 1
    config.system.ring_degree = 1

    # Data config
    config.data = ml_collections.ConfigDict()
    config.data.data_rootdir = ""
    config.data.sample_size = sample_size
    config.data.dilate_width = 11
    config.data.max_video_length = max_video_length
    config.data.fps = fps

    # Video model config
    config.video_model = ml_collections.ConfigDict()
    config.video_model.model_name = model_name
    config.video_model.transformer_path = transformer_path
    config.video_model.vae_path = vae_path
    config.video_model.lora_path = lora_path
    config.video_model.use_trimask = True
    config.video_model.zero_out_mask_region = False
    config.video_model.sampler_name = "DDIM_Origin"
    config.video_model.denoise_strength = denoise_strength
    config.video_model.negative_prompt = (
        "The video is not of a high quality, it has a low resolution. "
        "Watermark present in each frame. The background is solid. "
        "Strange body and strange trajectory. Distortion."
    )
    config.video_model.guidance_scale = guidance_scale
    config.video_model.num_inference_steps = num_inference_steps
    config.video_model.lora_weight = 0.55
    config.video_model.temporal_window_size = 85
    config.video_model.temproal_multidiffusion_stride = 16
    config.video_model.use_vae_mask = False
    config.video_model.stack_mask = False

    # Experiment config
    config.experiment = ml_collections.ConfigDict()
    config.experiment.run_seqs = ""
    config.experiment.matting_mode = "solo"
    config.experiment.save_path = "casper_outputs"
    config.experiment.skip_if_exists = False
    config.experiment.validation = False
    config.experiment.skip_unet = False
    config.experiment.mask_to_vae = False

    return config


def get_omnimatte_config(
    source_video_dir: str = "",
    sample_size: str = "384x672",
    max_video_length: int = 197,
    fps: int = 16,
    num_steps: int = 6000,
    batch_size: int = 16,
    seed: int = 43,
    device: str = "cuda",
    resegment: bool = True,
    erode_mask_width: int = 5,
    detail_transfer: bool = True,
) -> ml_collections.ConfigDict:
    """
    Create a configuration for the Omnimatte reconstruction model.

    Args:
        source_video_dir: Directory containing generated videos from Casper
        sample_size: Video size as "HxW" (e.g., "384x672")
        max_video_length: Maximum number of frames to process
        fps: Frames per second for output video
        num_steps: Number of optimization steps
        batch_size: Batch size for optimization
        seed: Random seed for reproducibility
        device: Device to run on ("cuda" or "cpu")
        resegment: Whether to refine masks using segmentation
        erode_mask_width: Width to erode masks by
        detail_transfer: Whether to transfer details to final output

    Returns:
        ConfigDict with all model settings
    """
    config = ml_collections.ConfigDict()

    # System config
    config.system = ml_collections.ConfigDict()
    config.system.low_gpu_memory_mode = False
    config.system.weight_dtype = torch.bfloat16
    config.system.seed = seed
    config.system.allow_skipping_error = False
    config.system.device = device

    # Data config
    config.data = ml_collections.ConfigDict()
    config.data.data_rootdir = ""
    config.data.sample_size = sample_size
    config.data.dilate_width = 11
    config.data.max_video_length = max_video_length
    config.data.fps = fps

    # Omnimatte config
    config.omnimatte = ml_collections.ConfigDict()
    config.omnimatte.source_video_dir = source_video_dir
    config.omnimatte.background_video_dir = ""
    config.omnimatte.resegment = resegment
    config.omnimatte.erode_mask_width = erode_mask_width
    config.omnimatte.log_dir = "omnimatte_logs"
    config.omnimatte.freq_log = 50
    config.omnimatte.freq_eval = 100_000

    config.omnimatte.rgb_module_type = "unet"
    config.omnimatte.rgb_lr = 5e-4
    config.omnimatte.alpha_module_type = "unet"
    config.omnimatte.alpha_lr = 1e-3

    config.omnimatte.batch_size = batch_size
    config.omnimatte.num_steps = num_steps
    config.omnimatte.lr_schedule_milestones = [500, 1_000, 2_000]
    config.omnimatte.lr_schedule_gamma = 0.1

    config.omnimatte.loss_recon_metric = "l2"
    config.omnimatte.loss_mask_super_metric = "l2"
    config.omnimatte.loss_weight_recon = 1.0
    config.omnimatte.loss_weight_alpha_reg_l0 = 0.075
    config.omnimatte.loss_weight_alpha_reg_l1 = 0.75
    config.omnimatte.loss_weight_alpha_reg_l0_k = 5.0
    config.omnimatte.loss_weight_mask_super = 10.0
    config.omnimatte.loss_weight_mask_super_ones = 0.1
    config.omnimatte.loss_weight_smoothness = 0.0

    config.omnimatte.loss_weight_alpha_reg_l0_steps = [1_000, 1_500]
    config.omnimatte.loss_weight_alpha_reg_l1_steps = [1_000, 1_500]
    config.omnimatte.loss_weight_mask_super_steps = [1_000]
    config.omnimatte.loss_weight_mask_super_ones_steps = []
    config.omnimatte.loss_weight_smoothness_steps = []

    config.omnimatte.loss_weight_alpha_reg_l0_gamma = 1.0
    config.omnimatte.loss_weight_alpha_reg_l1_gamma = 1.0
    config.omnimatte.loss_weight_mask_super_gamma = 0.0
    config.omnimatte.loss_weight_smoothness_gamma = 1.0
    config.omnimatte.loss_weight_mask_super_ones_gamma = 1.0

    config.omnimatte.detail_transfer = detail_transfer
    config.omnimatte.composite_order = "0,1"
    config.omnimatte.detail_transfer_transmission_thresh = 0.98
    config.omnimatte.detail_transfer_use_input_mask = True

    # Experiment config
    config.experiment = ml_collections.ConfigDict()
    config.experiment.run_seqs = ""
    config.experiment.matting_mode = "solo"
    config.experiment.save_path = "omnimatte_outputs"
    config.experiment.skip_if_exists = False
    config.experiment.validation = False
    config.experiment.skip_unet = False
    config.experiment.mask_to_vae = False

    return config
