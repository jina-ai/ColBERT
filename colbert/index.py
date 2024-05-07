# scripts for indexing the collection

import argparse

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


def main(args):

    with Run().context(RunConfig(nranks=1, experiment=args.experiment)):
        config = ColBERTConfig(
            nbits=2,
            kmeans_niters=20,
            root="experiments",
            doc_maxlen=args.doc_maxlen,
        )
        indexer = Indexer(
            checkpoint=args.checkpoint,
            config=config
        )
        indexer.index(name=args.index_name, collection=args.collection, overwrite="resume")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='liuqi6777/jina-colbert-v2-round3')
    parser.add_argument('--experiment', type=str, default='msmarco')
    parser.add_argument('--index_name', type=str, default='colbert.round3.nbit=2.index')
    parser.add_argument('--collection', type=str, default='data/MSMARCO/collection.tsv')
    parser.add_argument('--doc_maxlen', type=int, default=300)
    args = parser.parse_args()
    main(args)
