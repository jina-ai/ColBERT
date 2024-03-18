import argparse
import json
import pytrec_eval


def evaluate(rel_path, rank_file, pid2real=None):
    if pid2real:
        pid2real = json.load(open(pid2real))

    qids_to_relevant_passageids = {}
    dev_query_positive_id = {}

    with open(rel_path) as f:
        for line in f.readlines()[1:]:
            qid, pid, rel = line.strip().split()
            rel = int(rel)

            if qid not in dev_query_positive_id:
                dev_query_positive_id[qid] = {}

            dev_query_positive_id[qid][pid] = rel

            if qid not in qids_to_relevant_passageids:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(pid)
                
    qids_to_ranked_candidate_passages = {}
    prediction = {}
    with open(rank_file) as f:
        for line in f:
            qid, pid, rank = line.strip().split('\t')[:3]
            if int(rank) > 15:
                continue
            
            if pid2real:
                pid = pid2real[pid]
            
            if qid not in prediction:
                prediction[qid] = {}
            if qid != pid:
                prediction[qid][pid] = -int(rank)

            if qid not in qids_to_ranked_candidate_passages:
                qids_to_ranked_candidate_passages[qid] = []
            qids_to_ranked_candidate_passages[qid].append(pid)

    evaluator = pytrec_eval.RelevanceEvaluator(
            dev_query_positive_id, {'ndcg_cut', 'recall'})

    result = evaluator.evaluate(prediction)

    eval_query_cnt = 0
    ndcg = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]

    final_ndcg = ndcg / eval_query_cnt

    print(final_ndcg)
    return final_ndcg


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--qrels", type=str, required=True)
    args.add_argument("--ranking", type=str, required=True)
    args.add_argument("--pid2real", type=str, default=None)
    args = args.parse_args()
    evaluate(args.qrels, args.ranking, args.pid2real)