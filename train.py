"""Training entrypoint wiring together preprocessing and RL trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rl.buffer import ReplayBuffer
from src.rl.trainer import RLTrainer, TrainerConfig
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KOL RL agent")
    parser.add_argument("--config", default="config/rl_config.yaml", help="Path to RL config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Loading config from %s", args.config)
    buffer = ReplayBuffer()
    config = TrainerConfig()
    trainer = RLTrainer(config, buffer)
    LOGGER.info("Starting training loop")
    trainer.run()
    LOGGER.info("Training finished; checkpoint at %s", config.checkpoint_path)


if __name__ == "__main__":
    main()
