import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine', 
                 matryoshka=False, output_dim=None):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)

        # TODO: refactor this
        if matryoshka:
            self.matryoshka = True
            self.nesting_list = [int(x) for x in matryoshka.split(',')]
            self.accumulated = [sum(self.nesting_list[:i]) for i in range(len(self.nesting_list) + 1)]
            self.linear = nn.Linear(config.hidden_size, sum(self.nesting_list), bias=False)
        else:
            self.matryoshka = False
            self.nesting_list = self.accumulated = None
            self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.output_dim = output_dim or self.dim
        if self.output_dim and self.nesting_list:
            assert self.output_dim in self.nesting_list

        self.init_weights()

    def forward(self, Q, D):
        Q = self.query(*Q)
        D = self.doc(*D)
        scores = []
        if self.matryoshka:
            for i in range(len(self.nesting_list)):
                Q_ = Q[:, :, self.accumulated[i]:self.accumulated[i + 1]]
                Q_ = torch.nn.functional.normalize(Q_, p=2, dim=2)
                D_ = D[:, :, self.accumulated[i]:self.accumulated[i + 1]]
                D_ = torch.nn.functional.normalize(D_, p=2, dim=2)
                scores.append(self.score(Q_, D_))
        else:
            scores.append(self.score(Q, D))
        return scores

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        if self.matryoshka and self.output_dim:
            i = self.nesting_list.index(self.output_dim)
            Q = Q[:, :, self.accumulated[i]:self.accumulated[i + 1]]
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        if self.matryoshka and self.output_dim:
            i = self.nesting_list.index(self.output_dim)
            D = D[:, :, self.accumulated[i]:self.accumulated[i + 1]]
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
