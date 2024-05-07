# script to train ColBERT on MSMARCO triples

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


if __name__ == '__main__':

    with Run().context(RunConfig(nranks=4, experiment="msmarco", name="triples.train.round3.bs=32.nway=64.ib.distilled")):

        config = ColBERTConfig(
            lr=1e-5,
            warmup=20_000,
            bsize=32,
            maxsteps=400_000,
            nway=64,
            use_ib_negatives=True,
            doc_maxlen=160,
            attend_to_mask_tokens=False,
            root="experiments",
        )

        trainer = Trainer(
            triples="data/MSMARCO/colbertv2.train.json",
            queries="data/MSMARCO/queries.train.tsv",
            collection="data/MSMARCO/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(
            checkpoint="experiments/msmarco/none/triples.train.round2.bs=32.nway=64.ib.distilled/checkpoints/colbert-200000",
        )

        print(f"Saved checkpoint to {checkpoint_path}...")

## CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m colbert.train > logs/triples.train.round3.bs=32.nway=64.ib.distilled.log 2>&1 &
