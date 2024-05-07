set -e

prefix=$1

if [ ! $prefix ]; then
    prefix="jina-colbert"
fi

echo "Evaluating the retrieval results for the BEIR datasets using the model: ${prefix}"

for dataset in "arguana" "climate-fever" "dbpedia-entity" "fever" "fiqa" "hotpotqa" "nfcorpus" "nq" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"; do
    if [ ! -f "experiments/beir/colbert.retrieve/beir/${prefix}.${dataset}.nbit=2.index.ranking.tsv" ]; then
        continue
    fi
    echo "${dataset}"
    python -m utility.evaluate.evaluate_custom_rankings \
        --qrels data/BEIR/${dataset}/qrels.test.tsv \
        --pid2real data/BEIR/${dataset}/idx2pid.jsonl \
        --ranking experiments/beir/colbert.retrieve/beir/${prefix}.${dataset}".nbit=2.index.ranking.tsv"
done
