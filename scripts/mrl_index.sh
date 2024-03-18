set -e

dim=768

CUDA_VISIBLE_DEVICES="6" python -m colbert.index \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
  --checkpoint experiments/MSMARCO-psg/train.py/msmarco.psg.l2.mrl/checkpoints/colbert-200000.dnn \
  --dim 1760 --output_dim ${dim} --matryoshka "32,64,128,256,512,768" \
  --collection data/BEIR/nfcorpus/corpus.tsv \
  --index_root experiments/indexes/ --index_name nfcorpus.mrl.${dim}.l2 \
  --root experiments/ --experiment beir

CUDA_VISIBLE_DEVICES="6" python -m colbert.index_faiss \
  --index_root experiments/indexes/ --index_name nfcorpus.mrl.${dim}.l2 \
  --partitions 32768 --sample 0.3 \
  --root experiments/ --experiment beir

CUDA_VISIBLE_DEVICES="6" python -m colbert.retrieve \
  --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
  --queries data/BEIR/nfcorpus/queries.tsv \
  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
  --index_root experiments/indexes/ --index_name nfcorpus.mrl.${dim}.l2 \
  --checkpoint experiments/MSMARCO-psg/train.py/msmarco.psg.l2.mrl/checkpoints/colbert-200000.dnn \
  --dim 1760 --output_dim ${dim} --matryoshka "32,64,128,256,512,768" \
  --root experiments/ --experiment beir