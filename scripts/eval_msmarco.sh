#!/bin/bash

set -e

checkpoint="jinaai/jina-colbert-v1-en"
index_name="jina-colbert.nbit=2.index"

CUDA_VISIBLE_DEVICES=0 python -m colbert.index \
    --checkpoint $checkpoint \
    --experiment msmarco \
    --index_name $index_name \
    --collection data/MSMARCO/collection.tsv

CUDA_VISIBLE_DEVICES=0 python -m colbert.retrieve \
    --experiment msmarco \
    --index_name $index_name \
    --queries data/MSMARCO/queries.dev.small.tsv
