import logging
import warnings
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

def run_training():
    pass

@hydra.main(version_base=None, config_path='../configs')
def main(hydra_cfg: DictConfig) -> str:
    experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return run_training(hydra_cfg, experiment_path)


if __name__ == '__main__':
    main()


