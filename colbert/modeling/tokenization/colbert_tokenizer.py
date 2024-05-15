from omegaconf import DictConfig
from transformers import AutoTokenizer
from typing import List


class ColBERTTokenizer:
    def __init__(
            self, 
            model_name_or_checkpoint: str,
            query_token: str,
            doc_token: str,
            query_maxlen: int,
            doc_maxlen: int,
            attend_to_mask_tokens: bool  
        ):
        self.query_token, self.doc_token = query_token, doc_token
        self.query_maxlen, self.doc_maxlen = query_maxlen, doc_maxlen
        self.attend_to_mask_tokens = attend_to_mask_tokens

        self.tok = AutoTokenizer.from_pretrained(model_name_or_checkpoint)

        if not all(
            token in self.tok.all_special_tokens
            for token in (self.query_token, self.doc_token)
        ):
            # add tokens
            self.tok.add_special_tokens(
                {"additional_special_tokens": [self.query_token, self.doc_token]}
            )

    def tensorize(self, batch_text: List[str], mode: str):
        assert mode in ("query", "document"), mode

        # add marker
        marker = self.query_token if mode == "query" else self.doc_token
        batch_text = [marker + x for x in batch_text]

        # only do max_length padding in the "query aumentation" case
        padding = (
            "max_length"
            if (mode == "query" and self.attend_to_mask_tokens)
            else "longest"
        )
        obj = self.tok(
            batch_text,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            max_length=self.query_maxlen if mode == "query" else self.doc_maxlen,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        if mode == "query":
            # postprocess for the [MASK] augmentation
            ids[ids == self.tok.pad_token_id] = self.tok.mask_token_id

            if self.attend_to_mask_tokens:
                mask[ids == self.tok.mask_token_id] = 1
                assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        return ids, mask

    def __len__(self) -> int:
        return len(self.tok)
