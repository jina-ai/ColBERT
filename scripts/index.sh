set -e

dim=32

CUDA_VISIBLE_DEVICES="7" python -m colbert.index \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
  --checkpoint experiments/MSMARCO-psg/train.py/msmarco.psg.l2.${dim}/checkpoints/colbert-200000.dnn --dim ${dim} \
  --collection data/BEIR/nfcorpus/corpus.tsv \
  --index_root experiments/indexes/ --index_name nfcorpus.${dim}.l2 \
  --root experiments/ --experiment beir

CUDA_VISIBLE_DEVICES="7" python -m colbert.index_faiss \
  --index_root experiments/indexes/ --index_name nfcorpus.${dim}.l2 \
  --partitions 32768 --sample 0.3 \
  --root experiments/ --experiment beir

CUDA_VISIBLE_DEVICES="7" python -m colbert.retrieve \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
  --queries data/BEIR/nfcorpus/queries.tsv \
  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
  --index_root experiments/indexes/ --index_name nfcorpus.${dim}.l2 \
  --checkpoint experiments/MSMARCO-psg/train.py/msmarco.psg.l2.${dim}/checkpoints/colbert-200000.dnn --dim ${dim} \
  --root experiments/ --experiment beir

# python scripts/evaluate_custom_rankings.py \
#   --qrels data/BEIR/nfcorpus/qrels.test.tsv \
#   --pid2real data/BEIR/nfcorpus/idx2pid.jsonl \
#   --ranking /home/qliu/workspace/ColBERT/experiments/beir/retrieve.py/2024-02-03_03.48.04/ranking.tsv
