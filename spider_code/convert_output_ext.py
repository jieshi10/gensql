import json
from copy import deepcopy

FILENAME = r'output/qwen2-7b-chat-spider-full_db-fs8-dev'

INPUT_FILE = FILENAME + '.json'
OUTPUT_GOLD = FILENAME + '-gold.txt'
OUTPUT_PRED = FILENAME + '-pred.txt'

field = 'infer'

# use_ext = True
use_ext = False

with open(INPUT_FILE, encoding='utf-8') as f:
    data = json.load(f)


def get_dict_value(r: dict, field: str):
    if '.' in field:
        field = field.split('.')
        for k in field:
            try:
                k = int(k)
            except:
                pass
            r = r[k]
        return r
    else:
        return r[field]


def set_dict_value(r: dict, field: str, value):
    if '.' in field:
        field = field.split('.')
        for k in field[:-1]:
            try:
                k = int(k)
            except:
                pass
            if k not in r:
                r[k] = {}
            r = r[k]
        k = field[-1]
        try:
            k = int(k)
        except:
            pass
        r[k] = value
    else:
        r[field] = value


with open('benchmark/spider/dev.json', encoding='utf-8') as f:
    dev_data = json.load(f)

with open(OUTPUT_GOLD, 'w', encoding='utf-8') as fg, \
        open(OUTPUT_PRED, 'w', encoding='utf-8') as fp:
    for r, dev_r in zip(data, dev_data):
        if r is None:
            r = deepcopy(dev_r)
            r['output'] = r['query']
            set_dict_value(r, field, 'SELECT')
        assert r['question'] == dev_r['question']
        assert r['output'] == dev_r['query']
        assert r['db'] == dev_r['db_id']
        gold = r['output']
        pred = get_dict_value(r, field).replace('\n', ' ').replace('\t', ' ').strip()
        db = r['db']
        if use_ext:
            db += '_ext'
        fg.write(gold + '\t' + db + '\n')
        fp.write(pred + '\t' + db + '\n')