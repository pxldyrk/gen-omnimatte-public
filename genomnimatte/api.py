"""
High-level API for GenOmnimatte models.

Provides clean wrapper classes for:
- CasperModel: CogVideoX-based video inpainting
- OmnimatteModel: Omnimatte reconstruction for matte extraction
"""

import os
import sys
import gc
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

import torch
import numpy as np
from loguru import logger

# Add parent directory to path for imports
_package_root = Path(__file__).parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

from genomnimatte.config import get_casper_config, get_omnimatte_config


@dataclass
class InferenceResult:
    """Result from model inference."""
    output_path: str
    video_tensor: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class CasperModel:
    """
    CogVideoX-based video inpainting model (Casper).

    This model takes a video and a mask, and generates a new video with
    the masked region inpainted.

    Example:
        >>> model = CasperModel(model_path="path/to/CogVideoX-Fun-V1.5-5b-InP")
        >>> result = model.run(
        ...     video="input.mp4",
        ...     mask="mask.mp4",
        ...     output_dir="outputs/",
        ... )
    """

    def __init__(
        self,
        model_path: str,
        transformer_path: str = "",
        vae_path: str = "",
        lora_path: str = "",
        device: str = "cuda",
        gpu_memory_mode: str = "model_cpu_offload_and_qfloat8",
        seed: int = 43,
    ):
        """
        Initialize the Casper model.

        Args:
            model_path: Path to the pretrained CogVideoX model directory
            transformer_path: Optional path to custom transformer weights
            vae_path: Optional path to custom VAE weights
            lora_path: Optional path to LoRA weights
            device: Device to run on ("cuda" or "cpu")
            gpu_memory_mode: Memory optimization mode. Options:
                - "model_full_load": Load entire model to GPU
                - "model_cpu_offload": Offload to CPU after use
                - "model_cpu_offload_and_qfloat8": Offload + quantize to float8
                - "sequential_cpu_offload": Layer-by-layer offload (slowest, lowest memory)
            seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.transformer_path = transformer_path
        self.vae_path = vae_path
        self.lora_path = lora_path
        self.device = device
        self.gpu_memory_mode = gpu_memory_mode
        self.seed = seed

        self._pipeline = None
        self._vae = None
        self._generator = None
        self._config = None

    def _load_pipeline(self, config):
        """Load the inference pipeline."""
        # Import here to avoid loading heavy dependencies until needed
        from videox_fun.models import (
            AutoencoderKLCogVideoX,
            CogVideoXTransformer3DModel,
            T5EncoderModel,
            T5Tokenizer,
        )
        from videox_fun.pipeline import (
            CogVideoXFunPipeline,
            CogVideoXFunInpaintPipeline,
        )
        from videox_fun.utils.lora_utils import merge_lora
        from videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper
        from videox_fun.dist import set_multi_gpus_devices
        from diffusers import (
            CogVideoXDDIMScheduler,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
        )

        model_name = config.video_model.model_name
        weight_dtype = config.system.weight_dtype
        device = set_multi_gpus_devices(
            config.system.ulysses_degree, config.system.ring_degree
        )

        # Load transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=(
                torch.float8_e4m3fn
                if config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8"
                else weight_dtype
            ),
            use_vae_mask=config.video_model.use_vae_mask,
            stack_mask=config.video_model.stack_mask,
        ).to(weight_dtype)

        if config.video_model.transformer_path:
            logger.info(f"Loading transformer from: {config.video_model.transformer_path}")
            if config.video_model.transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(config.video_model.transformer_path)
            else:
                state_dict = torch.load(
                    config.video_model.transformer_path, map_location="cpu"
                )
            state_dict = state_dict.get("state_dict", state_dict)
            m, u = transformer.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded transformer: missing={len(m)}, unexpected={len(u)}")

        # Load VAE
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, subfolder="vae"
        ).to(weight_dtype)

        if config.video_model.vae_path:
            logger.info(f"Loading VAE from: {config.video_model.vae_path}")
            if config.video_model.vae_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(config.video_model.vae_path)
            else:
                state_dict = torch.load(config.video_model.vae_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            vae.load_state_dict(state_dict, strict=False)

        # Load tokenizer and text encoder
        tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=weight_dtype
        )

        # Load scheduler
        scheduler_map = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }
        scheduler_cls = scheduler_map[config.video_model.sampler_name]
        scheduler = scheduler_cls.from_pretrained(model_name, subfolder="scheduler")

        # Create pipeline
        if transformer.config.in_channels != vae.config.latent_channels:
            pipeline = CogVideoXFunInpaintPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )
        else:
            pipeline = CogVideoXFunPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )

        # Apply memory optimizations
        if config.system.gpu_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif config.system.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        generator = torch.Generator(device=device).manual_seed(config.system.seed)

        # Apply LoRA if specified
        if config.video_model.lora_path:
            pipeline = merge_lora(
                pipeline,
                config.video_model.lora_path,
                config.video_model.lora_weight,
                device=device,
            )

        return pipeline, vae, generator

    def load(self, **config_overrides) -> "CasperModel":
        """
        Load the model into memory.

        Args:
            **config_overrides: Override any config parameters

        Returns:
            self for method chaining
        """
        self._config = get_casper_config(
            model_name=self.model_path,
            transformer_path=self.transformer_path,
            vae_path=self.vae_path,
            lora_path=self.lora_path,
            device=self.device,
            gpu_memory_mode=self.gpu_memory_mode,
            seed=self.seed,
        )

        # Apply overrides
        for key, value in config_overrides.items():
            if "." in key:
                parts = key.split(".")
                obj = self._config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        self._pipeline, self._vae, self._generator = self._load_pipeline(self._config)
        logger.info("Casper model loaded successfully")
        return self

    def run(
        self,
        video: Union[str, torch.Tensor, np.ndarray],
        mask: Union[str, torch.Tensor, np.ndarray],
        output_dir: str,
        prompt: str = "",
        sample_size: str = "384x672",
        max_frames: int = 197,
        fps: int = 16,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        denoise_strength: Optional[float] = None,
        keep_fg_ids: List[int] = [-1],
    ) -> InferenceResult:
        """
        Run video inpainting.

        Args:
            video: Input video path or tensor (T, C, H, W)
            mask: Mask video path or tensor (T, 1, H, W)
            output_dir: Directory to save output videos
            prompt: Optional text prompt for generation
            sample_size: Output size as "HxW"
            max_frames: Maximum number of frames to process
            fps: Output video FPS
            num_inference_steps: Number of diffusion steps (default: 50)
            guidance_scale: CFG scale (default: 1.0)
            denoise_strength: Denoising strength (default: 1.0)
            keep_fg_ids: Foreground IDs to keep

        Returns:
            InferenceResult with output path and optional tensors
        """
        if self._pipeline is None:
            self.load()

        # Update config with runtime parameters
        self._config.data.sample_size = sample_size
        self._config.data.max_video_length = max_frames
        self._config.data.fps = fps
        self._config.experiment.save_path = output_dir

        if num_inference_steps is not None:
            self._config.video_model.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            self._config.video_model.guidance_scale = guidance_scale
        if denoise_strength is not None:
            self._config.video_model.denoise_strength = denoise_strength

        # Import utilities
        from videox_fun.utils.utils import (
            get_video_mask_input,
            save_videos_grid,
            save_inout_row,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Prepare video and mask
        video_length = self._config.data.max_video_length
        video_length = (
            int(
                (video_length - 1)
                // self._vae.config.temporal_compression_ratio
                * self._vae.config.temporal_compression_ratio
            )
            + 1
            if video_length != 1
            else 1
        )

        sample_size_tuple = tuple(map(int, sample_size.split("x")))

        if isinstance(video, str):
            # Load from data directory structure
            self._config.data.data_rootdir = str(Path(video).parent)
            input_video_name = Path(video).stem
            input_video, input_mask, loaded_prompt, _ = get_video_mask_input(
                input_video_name,
                sample_size=sample_size_tuple,
                keep_fg_ids=keep_fg_ids,
                max_video_length=video_length,
                temporal_window_size=self._config.video_model.temporal_window_size,
                data_rootdir=self._config.data.data_rootdir,
                use_trimask=self._config.video_model.use_trimask,
                dilate_width=self._config.data.dilate_width,
            )
            if not prompt:
                prompt = loaded_prompt
        else:
            input_video = video if isinstance(video, torch.Tensor) else torch.from_numpy(video)
            input_mask = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)

        # Run inference
        with torch.no_grad():
            sample = self._pipeline(
                prompt,
                num_frames=self._config.video_model.temporal_window_size,
                negative_prompt=self._config.video_model.negative_prompt,
                height=sample_size_tuple[0],
                width=sample_size_tuple[1],
                generator=self._generator,
                guidance_scale=self._config.video_model.guidance_scale,
                num_inference_steps=self._config.video_model.num_inference_steps,
                video=input_video,
                mask_video=input_mask,
                strength=self._config.video_model.denoise_strength,
                use_trimask=self._config.video_model.use_trimask,
                zero_out_mask_region=self._config.video_model.zero_out_mask_region,
                skip_unet=self._config.experiment.skip_unet,
                use_vae_mask=self._config.video_model.use_vae_mask,
                stack_mask=self._config.video_model.stack_mask,
            ).videos

        # Save output
        fg_str = "_".join([f"{i:02d}" for i in keep_fg_ids])
        output_name = f"output-fg={fg_str}"
        output_path = os.path.join(output_dir, f"{output_name}.mp4")

        if video_length == 1:
            from PIL import Image
            output_path = output_path.replace(".mp4", ".png")
            image = sample[0, :, 0].transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            Image.fromarray(image).save(output_path)
        else:
            save_videos_grid(sample, output_path, fps=fps)
            save_inout_row(
                input_video,
                input_mask,
                sample,
                output_path.replace(".mp4", "_comparison.mp4"),
                fps=fps,
            )

        return InferenceResult(
            output_path=output_path,
            video_tensor=sample,
            metadata={"prompt": prompt, "fps": fps},
        )

    def unload(self):
        """Unload model from memory."""
        del self._pipeline
        del self._vae
        del self._generator
        self._pipeline = None
        self._vae = None
        self._generator = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Casper model unloaded")


class OmnimatteModel:
    """
    Omnimatte reconstruction model for matte extraction.

    This model takes a source video and a generated video (from Casper),
    and reconstructs clean foreground/background layers with alpha mattes.

    Example:
        >>> model = OmnimatteModel()
        >>> result = model.run(
        ...     source_dir="path/to/sequence",
        ...     generated_dir="path/to/casper_outputs",
        ...     output_dir="outputs/",
        ... )
    """

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 43,
        num_steps: int = 6000,
        batch_size: int = 16,
    ):
        """
        Initialize the Omnimatte model.

        Args:
            device: Device to run on ("cuda" or "cpu")
            seed: Random seed for reproducibility
            num_steps: Number of optimization steps
            batch_size: Batch size for optimization
        """
        self.device = device
        self.seed = seed
        self.num_steps = num_steps
        self.batch_size = batch_size
        self._config = None

    def run(
        self,
        source_dir: str,
        generated_dir: str,
        output_dir: str,
        sequence_name: str,
        fg_id: int = -1,
        num_fgs: int = 1,
        sample_size: str = "384x672",
        max_frames: int = 197,
        fps: int = 16,
        resegment: bool = True,
        erode_mask_width: int = 5,
        detail_transfer: bool = True,
    ) -> InferenceResult:
        """
        Run omnimatte reconstruction.

        Args:
            source_dir: Directory containing source videos and masks
            generated_dir: Directory containing Casper-generated videos
            output_dir: Directory to save output mattes
            sequence_name: Name of the sequence to process
            fg_id: Foreground ID to process (-1 for all)
            num_fgs: Total number of foreground objects
            sample_size: Video size as "HxW"
            max_frames: Maximum number of frames to process
            fps: Output video FPS
            resegment: Whether to refine masks using segmentation
            erode_mask_width: Width to erode masks by
            detail_transfer: Whether to transfer details to final output

        Returns:
            InferenceResult with output path
        """
        import glob
        import torch.nn.functional as F
        import mediapy as media

        from omnimatte.optimization import OmnimatteOptimizer
        from omnimatte.utils import save_omnimatte, refine_mask, transfer_detail
        from videox_fun.utils.utils import get_video_mask_input, erode_video_mask

        # Create config
        self._config = get_omnimatte_config(
            source_video_dir=generated_dir,
            sample_size=sample_size,
            max_video_length=max_frames,
            fps=fps,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            seed=self.seed,
            device=self.device,
            resegment=resegment,
            erode_mask_width=erode_mask_width,
            detail_transfer=detail_transfer,
        )
        self._config.data.data_rootdir = source_dir
        self._config.experiment.save_path = output_dir

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        os.makedirs(output_dir, exist_ok=True)

        sample_size_tuple = tuple(map(int, sample_size.split("x")))

        # Load input video and mask
        input_video, input_mask, _, _ = get_video_mask_input(
            sequence_name,
            sample_size=sample_size_tuple,
            keep_fg_ids=[fg_id],
            max_video_length=max_frames,
            apply_temporal_padding=False,
            data_rootdir=source_dir,
            use_trimask=True,
            dilate_width=0,
        )

        input_video = input_video[0].permute(1, 0, 2, 3)  # (t, 3, h, w)
        input_mask = input_mask[0].permute(1, 0, 2, 3)  # (t, 1, h, w)

        def align_video(_video):
            _video = _video[: input_video.shape[0]].to(
                input_video.device, input_video.dtype
            )
            return F.interpolate(_video, size=input_video.shape[-2:], mode="bilinear")

        # Load generated video
        def read_casper_output(seq_name, fg_id):
            video_paths = [
                f
                for f in sorted(
                    glob.glob(
                        os.path.join(generated_dir, f"{seq_name}-fg={fg_id:02d}-*.mp4")
                    )
                )
                if not f.endswith("tuple.mp4")
            ]
            if not video_paths:
                raise FileNotFoundError(
                    f"No video found for {seq_name}, fg_id={fg_id} in {generated_dir}"
                )
            video_path = video_paths[-1]
            video = media.read_video(video_path).astype(float) / 255.0
            return torch.from_numpy(video).permute(0, 3, 1, 2)

        generated_video = read_casper_output(sequence_name, fg_id)
        generated_video = align_video(generated_video)

        if num_fgs > 1:
            mask_binary = torch.where(
                input_mask < 0.25, 1.0, 0.0
            ).to(input_mask.device, input_mask.dtype)
            solo_video = generated_video
            bg_video = read_casper_output(sequence_name, -1)
            bg_video = align_video(bg_video)
        else:
            mask_binary = torch.where(
                input_mask > 0.75, 1.0, 0.0
            ).to(input_mask.device, input_mask.dtype)
            solo_video = input_video
            bg_video = generated_video

        # Refine mask if requested
        if resegment:
            mask_np = mask_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)
            rgb_np = solo_video.detach().cpu().numpy().transpose(0, 2, 3, 1)
            mask_refined_np = refine_mask(rgb_np, mask_np)[:, None, :, :]
            mask_binary = torch.from_numpy(mask_refined_np).to(
                input_mask.device, input_mask.dtype
            )

        # Erode mask if requested
        if erode_mask_width:
            mask_np = mask_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)
            mask_np = (
                erode_video_mask(mask_np, erode_mask_width).astype(float) / 255.0
            )
            mask_np = mask_np.transpose(0, 3, 1, 2)
            mask_binary = torch.from_numpy(mask_np).to(
                input_mask.device, input_mask.dtype
            )

        # Run optimization
        logger.info(f"Running omnimatte optimization for {sequence_name}, fg_id={fg_id}")
        optimizer = OmnimatteOptimizer(
            self._config,
            XY=solo_video,
            Y=bg_video,
            init_mask=mask_binary,
            device=self.device,
            expname=f"{sequence_name}-fg={fg_id}",
        )
        optimization_outputs = optimizer.run()

        # Save outputs
        optimization_outputs["input_video"] = input_video
        save_omnimatte(
            optimization_outputs,
            output_dir,
            sequence_name,
            max(0, fg_id),
            fps=fps,
        )

        # Cleanup
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()

        save_dir = os.path.join(output_dir, sequence_name, f"fg{max(0, fg_id):02d}")

        # Run detail transfer if requested
        if detail_transfer:
            logger.info(f"Running detail transfer for {sequence_name}")
            transfer_detail(config=self._config, seq_name=sequence_name, num_fgs=num_fgs)

        return InferenceResult(
            output_path=save_dir,
            metadata={"sequence": sequence_name, "fg_id": fg_id},
        )
