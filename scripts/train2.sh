set -e

CUDA_VISIBLE_DEVICES=7 python -m colbert.train \
  --dim 128 --similarity l2 \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --maxsteps 200000 \
  --triples data/MSMARCO/triples.train.small.tsv \
  --root experiments/ --experiment MSMARCO-psg --run msmarco.psg.l2.128

CUDA_VISIBLE_DEVICES=7 python -m colbert.train \
  --dim 256 --similarity l2 \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --maxsteps 200000 \
  --triples data/MSMARCO/triples.train.small.tsv \
  --root experiments/ --experiment MSMARCO-psg --run msmarco.psg.l2.256

CUDA_VISIBLE_DEVICES=7 python -m colbert.train \
  --dim 512 --similarity l2 \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --maxsteps 200000 \
  --triples data/MSMARCO/triples.train.small.tsv \
  --root experiments/ --experiment MSMARCO-psg --run msmarco.psg.l2.512

