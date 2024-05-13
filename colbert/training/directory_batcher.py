import os
import ujson
from pathlib import Path
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.infra.config import ColBERTConfig

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial

from typing import List, Tuple

class TripletDataset(Dataset):
    def __init__(self, triples_path: Path | str, nway: int):
        self.nway = nway
        self.triples_path = triples_path
        self.paths: List = []
        if os.path.isdir(self.triples_path):
            self.paths = os.listdir(self.triples_path)
        elif os.path.isfile(self.triples_path):
            self.paths = [self.triples_path]
        else:
            raise ValueError(
                "triples_path should be a valid path to a file or directory"
            )

        # for *some* reproducibility
        self.paths.sort()
        self.data = []
        for path in self.paths:
            with open(os.path.join(self.triples_path, path), "r") as f:
                self.data.extend(f.readlines())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index].strip().split("\t")
        query, positive, negatives = data[0], data[1], data[2:]
        negatives = negatives[:self.nway-1]
        return (query, positive, negatives)


    # TODO: make lazy loading (if compatible with DataLoader batching)
    # May need to write dataloading logic by hand for BGE-M3-style batch size optimizations

    # def generate(self):
    #     for path in self.paths:
    #         with open(path, "r") as f:
    #             for line in f.readlines():
    #                 # TODO: Add logic to be able to read precomputed reranker supervision scores
    #                 if line.strip() == "\n":
    #                     continue
    #                 data = line.strip().split("\t")
    #                 # data : q, p, n_1, n_2, ...n_k
    #                 query, positive, negatives = data[0], data[1], data[2:]
    #                 # return the query and the first nway passages [1 positive, (nway - 1) negatives]
    #                 yield query, positive, negatives[:self.nway - 1]

    # def __iter__(self):
    #     return iter(self.generate())

    @staticmethod
    def collate_fn(
        query_tokenizer: QueryTokenizer,
        doc_tokenizer: DocTokenizer,
        data: List[Tuple[List[str], List[str], List[str]]],
    ):
        queries, positives, negatives = zip(*data)
        negatives_flattened : List[str] = [n for ns in negatives for n in ns]

        queries = query_tokenizer.tensorize(queries)
        positives = doc_tokenizer.tensorize(positives)
        negatives = doc_tokenizer.tensorize(negatives_flattened)

        return {
            "queries" : queries,
            "positives" : positives,
            "negatives" : negatives,
        }
    
    def create_dataloader(self, config : ColBERTConfig, *args, **kwargs):
        qt = QueryTokenizer(config)
        dt = DocTokenizer(config)
        collate_fn = partial(self.collate_fn, qt, dt)
        return DataLoader(self, *args, collate_fn=collate_fn, **kwargs)

if __name__ == "__main__":
    dataset = TripletDataset("/home/rohan/ColBERT/data/test_s3", nway=8)
    config = ColBERTConfig()
    config.checkpoint = "colbert-ir/colbertv1.9"
    qt = QueryTokenizer(config)
    dt = DocTokenizer(config)
    collate_fn = partial(dataset.collate_fn, qt, dt)
    dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)