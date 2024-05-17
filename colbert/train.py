from typing import Dict, List
from loguru import logger
import os
import hydra
import torch
from functools import partial
from collections.abc import Iterable
from omegaconf import DictConfig, OmegaConf

from lightning.fabric.fabric import Fabric
from lightning.pytorch import profilers, callbacks, strategies, loggers
from lightning import Trainer

from torch.utils.data import DataLoader

from colbert.data.collate import collate
from colbert.data.dataset import InputType, MultiDataset, _path_to_name
from colbert.data.utils import get_input_type
from colbert.modeling.tokenization import ColBERTTokenizer
from colbert.training.ColBERTLightning import ColBERTLightning


def module_class_constructor(module, cls_configs: DictConfig, singleton=False):
    objects = []
    for cls_name, options in cls_configs.items():
        object_cls = getattr(module, cls_name)
        objects.append(object_cls(**options))
    if singleton:
        if len(objects) > 1:
            raise ValueError(f"Can only specify one {module}, got {objects}")
        return objects[0]
    return objects


def run_training(config: DictConfig):
    logger.info(f"Config: \n{OmegaConf.to_yaml(config)}")

    logger.info("Building Tokenizer")
    # Make tokenizer
    tokenizer = ColBERTTokenizer(
        model_name_or_checkpoint=config.get("checkpoint", config.model.name),
        query_token=config.tokenizer.query_token,
        doc_token=config.tokenizer.doc_token,
        query_maxlen=config.tokenizer.query_maxlen,
        doc_maxlen=config.tokenizer.doc_maxlen,
        attend_to_mask_tokens=config.interaction.attend_to_mask_tokens,
    )
    new_vocab_size = len(tokenizer)

    # Make dataset/loader

    # required for dataset?
    fabric = Fabric()
    logger.info("Building MultiDataset")

    dataset_paths: List[str] = list(config.dataset.datasets)
    # resolve strings to InputType
    input_pattern_type_dict: Dict[str, InputType] = {
        pattern: InputType[itype]
        for pattern, itype in config.dataset.input_types.items()
    }

    input_type_dict = {
        _path_to_name(ds_path): get_input_type(input_pattern_type_dict, ds_path)
        for ds_path in dataset_paths
    }

    dataset = MultiDataset(
        bucket=config.dataset.s3_bucket,
        fabric=fabric,
        batch_size=config.dataloader.batch_size,
        input_type_dict=input_type_dict,
        datasets=dataset_paths,
        max_shards=config.dataset.max_shards,
        dialect=config.dataset.dialect,
    )

    collate_fn = partial(
        collate,
        tokenizer,
        input_type_dict,
        config.dataloader.num_negatives,
    )

    logger.info("Building DataLoader")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
    )

    resources = {k: v for k, v in config.resources.items() if k != "gpus"}
    gpus: int | List[int] = config.resources.gpus
    if isinstance(gpus, int):
        devices = gpus
    elif isinstance(gpus, Iterable):
        # make exactly the device ids specified visible
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(device_id) for device_id in gpus
        )
        logger.info(
            f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        # use "all" of them
        devices = len(gpus)
    else:
        raise ValueError(
            f"gpus {gpus} should be a number of devices to use, or a list of device ids"
        )

    constructed_trainer_kwargs = {}
    if config.callbacks:
        logger.info("Building Callbacks")
        train_callbacks: List[callbacks.Callback] = module_class_constructor(
            callbacks,
            config.callbacks,
            singleton=False,
        )
        constructed_trainer_kwargs["callbacks"] = train_callbacks
    if config.strategy:
        logger.info("Building Strategy")
        constructed_trainer_kwargs["strategy"] = module_class_constructor(
            strategies,
            config.strategy,
            singleton=True,
        )

    if config.profiler:
        logger.info("Building Profiler")
        constructed_trainer_kwargs["profiler"] = module_class_constructor(
            profilers,
            config.profiler,
            singleton=True,
        )

    if config.logger:
        # Make trainer (incl. callbacks, optimizer/scheduler)
        logger.info("Building Trainer")
        constructed_trainer_kwargs["logger"] = module_class_constructor(
            loggers,
            config.logger,
            singleton=True,
        )

    trainer = Trainer(
        devices=devices,
        **constructed_trainer_kwargs,
        **resources,
        **config.trainer_options,
    )

    # Make model
    logger.info("Building Model")
    model = ColBERTLightning(
        config=config,
        vocab_size=new_vocab_size,
    )

    logger.info("Setting CUDA float32_matmul_precision to medium")
    torch.set_float32_matmul_precision("medium")

    # Fit trainer
    logger.info("Training Model")
    trainer.fit(model=model, train_dataloaders=dataloader)


@hydra.main(version_base=None, config_path="../configs")
def main(hydra_cfg: DictConfig) -> str:
    return run_training(hydra_cfg)


if __name__ == "__main__":
    main()
