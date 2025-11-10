"""
Main script for training latent action model with LeRobot datasets.

This script uses Lightning CLI to train the DINO_LAM model with mixed LeRobot datasets.
"""

import torch
from genie.dataset import LightningLeRobotDataset
from genie.model import DINO_LAM
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('high')


def cli_main():
    """Main entry point for Lightning CLI."""
    cli = LightningCLI(
        DINO_LAM,
        LightningLeRobotDataset,
        seed_everything_default=42,
        save_config_callback=None,  # Disable auto-save of config
    )


if __name__ == "__main__":
    cli_main()
