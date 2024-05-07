#!/bin/bash

set -e

checkpoint="jinaai/jina-colbert-v1-en"
prefix=jina-colbert-v1.1

# This script evaluates a ColBERT model on the BEIR test sets.
for dataset in "arguana" "climate-fever" "dbpedia-entity" "fever" "fiqa" "hotpotqa" "nfcorpus" "nq" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" ; do

    if [ ! -d experiments/beir/indexes/${prefix}.${dataset}".nbit=2.index" ]; then
        CUDA_VISIBLE_DEVICES=0 python -m colbert.index \
            --checkpoint $checkpoint \
            --experiment beir \
            --index_name ${prefix}.${dataset}".nbit=2.index" \
            --collection data/BEIR/${dataset}/corpus.tsv
    fi

    if [ ! -f experiments/beir/colbert.retrieve/beir/${prefix}.${dataset}".nbit=2.index" ]; then
        CUDA_VISIBLE_DEVICES=0 python -m colbert.retrieve \
            --experiment beir \
            --index_name ${prefix}.${dataset}".nbit=2.index" \
            --queries data/BEIR/${dataset}/queries.tsv
    fi

    
done

## nohup sh scripts/eval_beir.sh > logs/eval_beir.log 2>&1 &
