# script to train ColBERT on MSMARCO triples

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__ == "__main__":

    with Run().context(
        RunConfig(
            nranks=1,
            experiment="msmarco",
            name="triples.train.shorttest.bs=8.nway=8.ib.distilled",
        )
    ):

        config = ColBERTConfig(
            lr=1e-5,
            warmup=20_000,
            bsize=4,
            nway=16,
            accumsteps=1,
            maxsteps=400_000,
            use_ib_negatives=True,
            distillation_alpha=1,
            query_maxlen=32,
            doc_maxlen=500,
            attend_to_mask_tokens=False,
            root="experiments",
            gpus="0",
            avoid_fork_if_possible=True,
        )

        trainer = Trainer(
            triples="/home/rohan/jina_colbert/data/MSMARCO/colbertv2.train.10k.nway=64.distilled.json",
            queries="/home/rohan/jina_colbert/data/MSMARCO/queries.train.tsv",
            collection="/home/rohan/jina_colbert/data/MSMARCO/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(
            checkpoint="colbert-ir/colbertv1.9",
        )

        print(f"Saved checkpoint to {checkpoint_path}...")
