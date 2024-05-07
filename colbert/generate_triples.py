from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert import Indexer
from colbert.ranking import Ranking
from colbert.utilities import Triples
from colbert.data.examples import Examples

from argparse import parser, Namespace

"""
TODO: This is a skeleton of the file, not a complete implementation
TODO: Replace all paths with argparsed inputs

Will need the following args:
-nranks
-experiment
-nbits
-root
-queries
-collection
-model_checkpoint
-ranking_topk
-triples_sampling_strategy
-triples_save_path
"""



def index() -> str:
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="/path/to/experiments",
        )
        indexer = Indexer(checkpoint="/path/to/checkpoint", config=config)
        indexer.index(name="msmarco.nbits=2", collection="/path/to/MSMARCO/collection.tsv")

def retrieve() -> Ranking:
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        # TODO: figure out how experiments works (probably to do with config args and data provenance)
        config = ColBERTConfig(
            root="/path/to/experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries("/path/to/MSMARCO/queries.dev.small.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
        return config.index_path
    
def generate_triples(ranking : Ranking, args : Namespace) -> Examples:
    # TODO: figure out positives format
    positives, depth = None, args.ranking_topk
    triples = Triples(ranking)
    triples.create(positives, depth)
    triples.save(args.triples_save_path)