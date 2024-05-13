import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lightning import Trainer, LightningModule
from lightning.fabric.fabric import Fabric

from typing import Dict, List, Union, Tuple, Any

from transformers import AdamW, get_linear_schedule_with_warmup

from colbert.data.dataset import InputType, MultiDataset
from colbert.modeling.colbert import ColBERT, ForkedPdb
from colbert.infra.config import ColBERTConfig


class ColBERTLightning(LightningModule):
    def __init__(self, config: ColBERTConfig):
        super().__init__()
        self.config = config
        self.colbert = ColBERT(name=config.checkpoint, colbert_config=self.config)

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
        # Tokenizer outputs: Tuple(input_ids, attention_mask)
        # b x q
        queries, query_mask = batch["queries"]
        # b x d
        positives, positive_mask = batch["positives"]
        # (b * n) x d
        negatives, negative_mask = batch["negatives"]

        # TODO: Should I replace self.colbert .query() and .doc() logic?
        # probably not, want to leave colbert as alone as possible so model can be loaded directly
        # for unchanged indexing/search logic

        ForkedPdb().set_trace()
        # b x q x h
        queries = self.colbert.query(queries, query_mask)
        # b x d x h
        positives = self.colbert.doc(positives, positive_mask)
        # b x b x d x h
        positives = positives.unsqueeze(1).repeat(1, positives.size(0), 1, 1)
        # (b * n) x d x h
        negatives = self.colbert.doc(negatives, negative_mask)
        # b x n x d x h
        negatives = negatives.reshape(
            positives.size(0), -1, negatives.size(1), negatives.size(2)
        )
        # b x (b + n) x d x h
        print(f"{negatives.shape} {positives.shape}")
        documents = torch.cat((positives, negatives), dim=1)
        # By construction the diagonal documents are the relevant ones
        labels = torch.arange(documents.size(0), dtype=torch.long)

        B_q, Q, H_q = queries.shape
        B_d, N, D, H_d = documents.shape

        assert B_q == self.config.bsize
        assert B_q == B_d
        assert H_q == H_d
        assert N == B_q + self.config.nway

        # Dimensions: b = bsize, h = hidden_dim, n = nway, q = max_qlen, d = max_dlen

        # expand dims to do token interaction (similarity metric)
        # queries : b x 1 x q x 1 x h
        # documents : b x n x 1 x d x h
        queries = queries[:, None, :, None, :]
        documents = documents[:, :, None, :, :]

        if self.config.similarity == "l2":
            # all_tok_scores : b x n x q x d
            all_tok_scores = -1 * (queries - documents).pow(2).sum(-1)
        else:
            raise ValueError("cosine similarity not supported yet")

        # -> max over max_dlen dimension
        # max_qtok_scores : b x (b + n) x q
        max_qtok_scores = all_tok_scores.max(-1).values
        # -> sum over max_qlen dimension
        # scores : b x (b + n)
        scores = max_qtok_scores.sum(-1)

        # compute loss on scores
        loss = torch.nn.CrossEntropyLoss()(scores, labels)
        self.log({"train_loss": loss})
        return loss


if __name__ == "__main__":
    from colbert.training.directory_batcher import TripletDataset
    import lightning as L

    config = ColBERTConfig(
        checkpoint="colbert-ir/colbertv1.9",
        warmup=10,
    )

    dataset_types = {
        "*": InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
    }

    fabric = Fabric()

    dataset = MultiDataset(
        bucket="embedding-datasets",
        fabric=fabric,
        batch_size=16,
        input_type_dict=dataset_types,
        datasets=[
            "en/triplets-multiple-negatives/msmarco-bge",
            "en/triplets-multiple-negatives/nq-bge",
        ],
        max_shards=1,
        dialect="tsv",
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.bsize,
        num_workers=0,
    )

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        fast_dev_run=True,
    )

    model = ColBERTLightning(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
    )
