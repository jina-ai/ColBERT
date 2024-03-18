CUDA_VISIBLE_DEVICES=7 python -m colbert.train \
  --dim 64 --similarity l2 --matryoshka "32,64,128,256,512,768" \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --maxsteps 200000 \
  --triples data/MSMARCO/triples.train.small.tsv \
  --root experiments/ --experiment MSMARCO-psg --run msmarco.psg.l2.mrl