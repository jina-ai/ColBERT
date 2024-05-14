import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lightning import Trainer, LightningModule
from lightning.fabric.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from typing import Dict, List, Union, Tuple, Any

from transformers import AdamW, get_linear_schedule_with_warmup

from colbert.data.dataset import InputType, MultiDataset
from colbert.modeling.colbert import ColBERT, ForkedPdb
from colbert.infra.config import ColBERTConfig
from colbert.data.collate import collate
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer

from functools import partial
from loguru import logger


class ColBERTLightning(LightningModule):
    def __init__(self, config: ColBERTConfig):
        super().__init__()
        self.config = config
        self.colbert = ColBERT(name=config.checkpoint, colbert_config=self.config)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.colbert.parameters()),
            lr=self.config.lr,
            eps=1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup,
            num_training_steps=self.config.maxsteps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
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

        if self.config.similarity == "cosine":
            # vectors are normalized along h dim by colbert.query() and doc()
            # so cosine similarity is just dot product
            # any interaction involving a masked token will be 0 
            # (and therefore not contribute, as desired)
            # all_tok_scores : b x n x q x d
            all_tok_scores = torch.einsum("bXqXh,bnXdh->bnqd", queries, documents)
        else:
            raise ValueError("l2 similarity not supported yet")
        
        # if self.config.similarity == "l2":
        #     # Negative so that max takes smallest l2 dist
        #     # all_tok_scores : b x n x q x d
        #     # TODO: is this even correct for masked tokens? 
        #     # is the l2 distance with 0's the worst one can do?
        #     # No. that distance is 1, but e.g. an antipode, would be 2.
        #     all_tok_scores = -1 * (queries - documents).pow(2).sum(-1)

        # -> max over max_dlen dimension
        # max_qtok_scores : b x (b + n) x q
        max_qtok_scores = all_tok_scores.max(-1).values
        # -> sum over max_qlen dimension
        # scores : b x (b + n)
        scores = max_qtok_scores.sum(-1)

        # compute loss on scores
        loss = torch.nn.CrossEntropyLoss()(scores, labels)
        self.log("train_loss", loss.item(), on_step=True, prog_bar=True)
        return loss

if __name__ == "__main__":
    from colbert.training.directory_batcher import TripletDataset
    import lightning as L

    config = ColBERTConfig(
        checkpoint="colbert-ir/colbertv1.9",
        warmup=10,
        similarity="cosine",
        bsize=16,
    )

    dataset_types = {
        "*": InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
        # "*": InputType.PAIR,
    }

    # I'm sure this will bite me in the ass in the future...
    fabric = Fabric()

    dataset = MultiDataset(
        bucket="embedding-datasets",
        fabric=fabric,
        batch_size=config.bsize,
        input_type_dict=dataset_types,
        datasets=[
            # "en/pairs_dedup/msmarco"
            # "en/triplets-multiple-negatives/msmarco-bge", # multiple negatives no scores
            "en/triplets-multiple-negatives/nq-bge", # multiple negatives no scores
        ],
        max_shards=1,
        dialect="tsv",
    )

    query_tokenizer = QueryTokenizer(config)
    doc_tokenizer = DocTokenizer(config)

    collate_fn = partial(
        collate,
        config,
        query_tokenizer,
        doc_tokenizer,
        dataset_types,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.bsize,
        num_workers=0,
        collate_fn=collate_fn,
    )

    wandb_logger = WandbLogger(project="jina-colbert")

    torch.set_float32_matmul_precision('medium')

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
        fast_dev_run=64,
        max_steps=1024,
        logger=wandb_logger,
    )

    model = ColBERTLightning(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
    )
