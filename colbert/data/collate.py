from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization import ColBERTTokenizer
from colbert.data.utils import lookahead
from colbert.data.dataset import InputType
from colbert.utils.utils import zipstar


def collate(
    tokenizer: ColBERTTokenizer,
    input_type_dict: Dict[str, InputType],
    num_negatives: int,
    batch: List[Tuple[str, Tuple[List[str], Union[List[float], None]]]],
) -> Dict[str, Union[Tuple[Tensor, Tensor], Tensor]]:
    
    # data instances are always of the form (dataset_name, (out, scores))
    dataset_names, text_items_and_scores = zipstar(batch)
    text_items, scores = zipstar(text_items_and_scores)

    batch_output = {}

    # Scores are non-none
    if None not in scores:
        batch_output["scores"] = scores

    # Invariant: assumes dataset_name in dict and all names in batch are same
    # Should be the case according to MultiDataset's batch logic
    assert set((dataset_names[0],)) == set(dataset_names), dataset_names
    dataset_name = dataset_names[0]
    # Either the dataset is in the dictionary or it is globbed with "*"
    input_type: InputType = input_type_dict.get(dataset_name, input_type_dict["*"])

    # tokenizers return (input_ids, attention_masks)
    batch_output["queries"] = tokenizer.tensorize(
        [splitline[0] for splitline in text_items],
        mode="query",
    )

    documents: List[str] = [splitline[1] for splitline in text_items]
    bsize = len(documents)

    use_negatives: bool = input_type in (
        InputType.TRIPLET,
        InputType.SCORED_TRIPLET,
        InputType.MULTIPLE_NEGATIVES,
        InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
    )

    if use_negatives:
        # negatives : bsize x nway
        negatives: List[List[str]] = [
            splitline[2 : num_negatives + 2] for splitline in text_items
        ]

        # must flatten them for doc tokenizer
        # negatives_flattened : (bsize * nway) x 1
        negatives_flattened: List[str] = [neg for negs in negatives for neg in negs]

        documents.extend(negatives_flattened)

    # (b + (b * n)) x max_doc_len
    # first b are positives, rest are negatives
    document_tokens, document_masks = tokenizer.tensorize(documents, mode="document")

    batch_output["positives"] = (document_tokens[:bsize], document_masks[:bsize])

    if use_negatives:
        batch_output["negatives"] = (document_tokens[bsize:], document_masks[bsize:])

    return batch_output
