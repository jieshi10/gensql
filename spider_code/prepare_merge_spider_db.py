import glob
import sqlite3

from sentence_transformers import SentenceTransformer, util

import json
from tqdm import tqdm


embedder = SentenceTransformer('./pretrained/bge-large-en-v1.5')


db = []
for db_path in glob.glob('benchmark/spider/database/*'):
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    tables = []
    with sqlite3.connect(f'{db_path}/{db_name}.sqlite') as conn:
        conn.text_factory = lambda x: str(x, 'latin1')
        c = conn.cursor()
        c.execute("select name, sql from sqlite_master where type='table';")
        for tbl_name, sql in c:
            if tbl_name == 'sqlite_sequence':
                continue
            # print(tbl_name)
            # print(sql)
            tables.append({
                'name': tbl_name,
                'sql': sql,
            })
            # print('=' * 30)
    db.append({
        'name': db_name,
        'tables': tables,
    })
print('total db:', len(db))


# x_123 => x
def get_raw_name(s: str):
    s = s.lower()
    if '_' in s and s.split('_')[-1].isnumeric():
        return s[:s.rindex('_')]
    else:
        return s


# similarity between two table collections
def calc_sim(tables1, tables2):
    x_embeds = []
    for x in tables1:
        x_name = x['name']
        x_raw_name = get_raw_name(x_name)
        x_sql = x['sql']
        x_embeds += [x_raw_name, x_sql]
    y_embeds = []
    for y in tables2:
        y_name = y['name']
        y_raw_name = get_raw_name(y_name)
        y_sql = y['sql']
        y_embeds += [y_raw_name, y_sql]
        if y_raw_name in x_embeds or y_sql in x_embeds:
            return 1
    x_embeds = embedder.encode(x_embeds, normalize_embeddings=True, convert_to_tensor=True)
    y_embeds = embedder.encode(y_embeds, normalize_embeddings=True, convert_to_tensor=True)
    sim = util.cos_sim(x_embeds, y_embeds)
    max_sim = 0
    for i in range(len(tables1)):
        for j in range(len(tables2)):
            s = (0.5*sim[2*i][2*j] + 0.5*sim[2*i+1][2*j+1]).item()
            max_sim = max(max_sim, s)
    return max_sim


res = []
for x in tqdm(db):
    y = {
        'name': f"{x['name']}_ext",
        'tables': x['tables'][:],
        'original': [x['name']],
    }
    for z in db:
        s = calc_sim(y['tables'], z['tables'])
        if s < 0.75:
            y['tables'] += z['tables']
            y['original'].append(z['name'])
    res.append(y)

for d in res:
    print(d['name'], len(d['tables']))

print(sum([len(d['tables']) for d in db]) / len(db))
print(sum([len(d['tables']) for d in res]) / len(res))

with open('spider_merged_schema_3.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)