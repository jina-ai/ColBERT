# script to retrieve the documents for a given query set

import argparse

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


def main(args):

    args.ranking = f"{args.index_name}.ranking.tsv"

    with Run().context(RunConfig(nranks=1, experiment=args.experiment, name=args.experiment)):

        config = ColBERTConfig(
            root="experiments",
            overwrite=True,
            query_maxlen=args.query_maxlen,
        )
        searcher = Searcher(index=args.index_name, config=config)
        queries = Queries(args.queries)
        ranking = searcher.search_all(queries, k=1000)
        ranking.save(args.ranking)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='msmarco')
    parser.add_argument('--index_name', type=str, default='colbert.round3.nbit=2.index')
    parser.add_argument('--queries', type=str, default='data/MSMARCO/queries.dev.small.tsv')
    parser.add_argument('--query_maxlen', type=int, default=32)
    args = parser.parse_args()
    main(args)
