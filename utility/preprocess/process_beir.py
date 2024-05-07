import ujson as json
import os
from tqdm import tqdm


datasets = []

for v in sorted(os.listdir('data/beir/original/qrels')):
    datasets.append(v.split('_')[0])
            
print(datasets)


for dataset in datasets:
    print(f'processing {dataset}...')
    os.makedirs(f'data/beir/{dataset}', exist_ok=True)
    
    idx2pid = {}
    
    os.system(f'cp data/beir/original/qrels/{dataset}_test.tsv data/beir/{dataset}/qrels.test.tsv')
        
    with open(f'data/beir/original/corpus/{dataset}.jsonl') as fin, open(f'data/beir/{dataset}/corpus.tsv', 'w') as fout:
        for idx, line in tqdm(enumerate(fin)):
            item = json.loads(line.strip())
            _id = str(item['docid'])
            idx2pid[str(idx)] = _id
            text = item['text'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            if item['title']:
                title = item['title'].replace('\n', ' ').replace('\t', ' ')
                text = title + ' ' + text
            fout.write(_id + '\t' + text + '\n')
            
    with open(f'data/beir/{dataset}/idx2pid.jsonl', 'w') as fout:
        json.dump(idx2pid, fout)

    with open(f'data/beir/original/queries/{dataset}.jsonl') as fin, open(f'data/beir/{dataset}/queries.tsv', 'w') as fout:
        for line in fin:
            item = json.loads(line.strip())
            _id = str(item['query_id'])
            text = item['query'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            fout.write(_id + '\t' + text + '\n')
