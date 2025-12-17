"""
GenOmnimatte: Generative Omnimatte for video object removal and matting.

This package provides a clean API for:
- CogVideoX-based video inpainting (CasperModel)
- Omnimatte reconstruction for matte extraction (OmnimatteModel)

Example usage:
    from genomnimatte import CasperModel, OmnimatteModel

    # Video inpainting with CogVideoX
    casper = CasperModel(model_path="path/to/model")
    casper.run(video_path="input.mp4", mask_path="mask.mp4", output_path="output/")

    # Omnimatte reconstruction
    omnimatte = OmnimatteModel()
    omnimatte.run(video_path="input.mp4", generated_video_path="generated.mp4", output_path="output/")
"""

from genomnimatte.api import CasperModel, OmnimatteModel
from genomnimatte.config import get_casper_config, get_omnimatte_config

__version__ = "0.1.0"
__all__ = [
    "CasperModel",
    "OmnimatteModel",
    "get_casper_config",
    "get_omnimatte_config",
]
