import logging
import warnings
from loguru import logger
import hydra
from functools import partial
from omegaconf import DictConfig, OmegaConf

from lightning.fabric.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from colbert.data.collate import collate
from colbert.data.dataset import MultiDataset
from colbert.modeling.tokenization import ColBERTTokenizer


def run_training(config: DictConfig):

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
    # Make dataset/loader

    # required for dataset?
    fabric = Fabric()
    logger.info("Building MultiDataset")
    dataset = MultiDataset(
        bucket=config.dataset.s3_bucket,
        fabric=fabric,
        batch_size=config.dataloader.batch_size,
        input_type_dict=config.datasets,
        datasets=config.datasets.keys(),
        max_shards=config.datasets.max_shards,
        dialect=config.datasets.dialect,
    )

    collate_fn = partial(
        collate,
        tokenizer,
        config.datasets,
        config.dataloader.num_negatives,
    )

    logger.info("Building DataLoader")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
    )

    # Make trainer (incl. callbacks, optimizer/scheduler)
    logger.info("Building Trainer")

    wandb_logger = WandbLogger(
        project=config.wandb.project,
    )

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=config.resources.accelerator,
        devices=config.resources.devices,
        precision=config.resources.precision,
    )

    # Make model
    model = ColBERTLightning(
        #TODO
        None
    )
    # Fit trainer


@hydra.main(version_base=None, config_path="../configs")
def main(hydra_cfg: DictConfig) -> str:
    return run_training(hydra_cfg)

if __name__ == "__main__":
    main()
