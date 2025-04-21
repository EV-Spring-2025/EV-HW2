from pathlib import Path

from omegaconf import OmegaConf

from src.trainer import Trainer


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    trainer = Trainer(
        data_dir=Path(config.data.dir, config.data.sport_type),
        output_dir=Path(config.output.dir, config.data.sport_type),
    )
    trainer.fit()
