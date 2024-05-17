import torch
from transformers import AutoTokenizer

from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from colbert.parameters import DEVICE


class DocTokenizer:
    def __init__(self, config: ColBERTConfig):

        self.tok = AutoTokenizer.from_pretrained(config.model_name)

        if not all(
            token in self.tok.all_special_tokens
            for token in (config.query_token, config.doc_token)
        ):
            # add tokens
            self.tok.add_special_tokens(
                {"additional_special_tokens": [config.query_token, config.doc_token]}
            )

        self.config = config
        self.doc_maxlen = config.doc_maxlen

        self.D_marker_token, self.D_marker_token_id = config.doc_token, self.tok.convert_tokens_to_ids(config.doc_token)
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [
            self.tok.tokenize(x, add_special_tokens=False).to(DEVICE)
            for x in batch_text
        ]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [
            self.sep_token_id
        ]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [D] marker
        batch_text = [". " + x for x in batch_text]

        padding = "max_length" if self.config.attend_to_mask_tokens else "longest"
        obj = self.tok(
            batch_text,
            padding=padding,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.doc_maxlen,
        ).to(DEVICE)

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
