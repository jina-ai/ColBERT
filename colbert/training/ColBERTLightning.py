import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW

from lightning import Trainer, LightningModule
from lightning.fabric.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from typing import Dict, List, Union, Tuple, Any

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_scheduler,
)

from omegaconf import DictConfig

from colbert.data.dataset import InputType, MultiDataset
from colbert.modeling.colbert import ColBERT, ForkedPdb
from colbert.infra.config import ColBERTConfig
from colbert.data.collate import collate
from colbert.modeling.tokenization import ColBERTTokenizer

from functools import partial
from loguru import logger


class ColBERTLightning(LightningModule):
    def __init__(self, config: DictConfig, vocab_size: int):
        super().__init__()
        self.config = config

        # initialize model
        checkpoint: str | None = config.model.checkpoint
        name_or_path = checkpoint if checkpoint is not None else config.model.name

        # should be able to get away with just passing the name/checkpoint
        # from which to load model weights
        self.colbert = ColBERT(name=name_or_path, colbert_config=None)

        # Extend colbert word embeddings to have query and doc markers
        # resize_token_embeddings is idempotent if given the same vocabulary size
        # so it doesn't matter if the embeddings are the correct size already,
        # e.g. due to loading from a checkpoint
        try:
            # This should be sufficient for most *normal* models
            self.colbert.model.base_model.resize_token_embeddings(vocab_size)
        except:
            # Our flash attention implementation of XLM-roberta hasn't implemented a number of methods,
            # despite them probably being identical to PreTrainedModel
            old_embs = self.colbert.model.base_model.embeddings.word_embeddings
            new_embs = torch.nn.Embedding(
                vocab_size,
                old_embs.embedding_dim,
                device=old_embs.weight.device,
                dtype=old_embs.weight.dtype,
                padding_idx=old_embs.padding_idx,
            )
            # PreTrainedModel does this init, but our model doesn't seem to change
            # anything from the original initialization
            self.colbert.model.base_model._init_weights(new_embs)

            # copy old weights over and replace old word embedding
            # detach gets around an error about modifying tensors that require_grad,
            # but still modifies the same underlying data.
            new_embs.weight.detach()[
                : old_embs.num_embeddings, :
            ] = old_embs.weight.detach()
            self.colbert.model.base_model.embeddings.word_embeddings = new_embs

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.colbert.parameters())
        match self.config.hyperparameters.optimizer.name:
            case "AdamW":
                optimizer = AdamW(
                    params=parameters, **self.config.hyperparameters.optimizer.options
                )
            case _:
                raise NotImplemented(
                    "Optimizer {self.config.hyperparameters.optimizer.name} is not supported"
                )

        scheduler = get_scheduler(
            optimizer=optimizer,
            **self.config.hyperparameters.scheduler,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(
        self,
        batch: Dict[
            str, Tuple[Tensor, Tensor]
        ],  # TODO: ammend typing for when we support supervised scores
        batch_idx: int,
    ) -> Tensor:
        # Dimensions commented above variable:
        # b = bsize, h = hidden_dim, n = nway, q = max_qlen, d = max_dlen

        # Tokenizer output format: Tuple(input_ids, attention_mask)
        # b x q
        queries, query_mask = batch["queries"]
        # b x d
        positives, positive_mask = batch["positives"]
        # Note that colbert.query() and doc() normalize along the h dim
        # b x q x h
        queries = self.colbert.query(queries, query_mask)
        # b x d x h
        positives = self.colbert.doc(positives, positive_mask)
        # b x b x d x h
        # For in-batch negatives, stack the positives so that each query
        # computes an interaction with each positive
        positives = positives.unsqueeze(0).repeat(positives.size(0), 1, 1, 1)

        # TODO: refactor this so that positives + negatives are run through colbert.doc() together? 
        # (then split and reformatted to be recombined)
        # currently w/ negatives has 3 sequential calls of ColBERT
        if "negatives" in batch:
            # expect the flattened negatives from the batch to be compatible with colbert.doc()
            # (b * n) x d
            negatives, negative_mask = batch["negatives"]
            # (b * n) x d x h
            negatives = self.colbert.doc(negatives, negative_mask)
            # b x n x d x h
            negatives = negatives.reshape(
                positives.size(0), -1, negatives.size(1), negatives.size(2)
            )
            # b x (b + n) x d x h
            documents = torch.cat((positives, negatives), dim=1)
        else:
            documents = positives
        # By construction the diagonal documents are the relevant ones
        labels = torch.arange(
            documents.size(0),
            dtype=torch.long,
            device=self.device,
        )
        # expand dims to do token interaction (similarity metric)
        # queries : b x 1 x q x 1 x h
        # documents : b x n x 1 x d x h
        queries = queries[:, None, :, None, :]
        documents = documents[:, :, None, :, :]

        if self.config.interaction.similarity == "cosine":
            # vectors are normalized along h dim by colbert.query() and doc()
            # so cosine similarity is just dot product
            # any interaction involving a masked token will be 0
            # (and therefore not contribute, as desired)
            # all_tok_scores : b x n x q x d
            all_tok_scores = torch.einsum("bXqXh,bnXdh->bnqd", queries, documents)
        else:
            raise ValueError("l2 similarity not supported yet")

        # -> max over max_dlen dimension
        # max_qtok_scores : b x (b + n) x q
        max_qtok_scores = all_tok_scores.max(-1).values
        # -> sum over max_qlen dimension
        # scores : b x (b + n)
        scores = max_qtok_scores.sum(-1)

        # compute loss on scores with/out supervision
        if "scores" in batch and self.config.hyperparameters.loss == "KLDiv":
            raise NotImplemented("Score supervision not supported yet")
        else:
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        self.log("train_loss", loss.item(), on_step=True, prog_bar=True)
        return loss


if __name__ == "__main__":
    from colbert.training.directory_batcher import TripletDataset
    import lightning as L

    config = ColBERTConfig(
        # model_name="jinaai/jina-xlm-roberta-base-8k",
        model_name="colbert-ir/colbertv1.9",
        checkpoint=None,
        warmup=200,
        similarity="cosine",
        bsize=32,
        lr=3e-6,
    )

    dataset_types = {
        # "*": InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
        "*": InputType.PAIR,
    }

    # I'm sure this will bite me in the ass in the future...
    fabric = Fabric()

    dataset = MultiDataset(
        bucket="embedding-datasets",
        fabric=fabric,
        batch_size=config.bsize,
        input_type_dict=dataset_types,
        datasets=[
            "en/pairs_dedup/msmarco"
            # "en/triplets-multiple-negatives/msmarco-bge", # multiple negatives no scores
            # "en/triplets-multiple-negatives/nq-bge",  # multiple negatives no scores
        ],
        max_shards=1,
        dialect="tsv",
    )

    tokenizer = ColBERTTokenizer(config)

    collate_fn = partial(
        collate,
        config,
        tokenizer,
        dataset_types,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.bsize,
        num_workers=0,
        collate_fn=collate_fn,
    )

    wandb_logger = WandbLogger(project="jina-colbert")

    torch.set_float32_matmul_precision("medium")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[1],
        # accelerator="cpu",
        precision="16-mixed",  # note that 16-true causes training instability immediately (NaN weights)
        # fast_dev_run=64,
        max_steps=4096,
        logger=wandb_logger,
    )

    model = ColBERTLightning(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
    )
