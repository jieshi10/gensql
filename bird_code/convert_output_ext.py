import json
from copy import deepcopy

FILENAME = r'output/qwen2-7b-chat-bird-full_db-fs8-dev'

INPUT_FILE = FILENAME + '.json'
OUTPUT_GOLD = FILENAME + '-gold.sql'
OUTPUT_PRED = FILENAME + '-pred.json'
OUTPUT_DEV = FILENAME + '-dev.json'

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


with open('benchmark/BIRD/dev/dev.json', encoding='utf-8') as f:
    dev_data = json.load(f)

out_data = []
out_dev_data = []
assert len(data) == len(dev_data)
for i, (r, dev_r) in enumerate(zip(data, dev_data)):
    if r is None:
        r = deepcopy(dev_r)
        r['db'] = r['db_id']
        r['output'] = r['SQL']
        r['question'] += ' Hint: ' + r['evidence']
        set_dict_value(r, field, 'SELECT')
    out_data.append(r)
    out_dev_data.append(dev_r)
    assert i == dev_r['question_id']
    assert r['db'] == dev_r['db_id']
    assert r['output'] == dev_r['SQL']
    assert r['question'] == dev_r['question'] + ' Hint: ' + dev_r['evidence']

with open(OUTPUT_GOLD, 'w', encoding='utf-8') as fg, \
        open(OUTPUT_PRED, 'w', encoding='utf-8') as fp:
    res = {}
    assert len(out_data) == len(out_dev_data)
    for i, (r, dev_r) in enumerate(zip(out_data, out_dev_data)):
        assert i == dev_r['question_id']
        assert r['db'] == dev_r['db_id']
        assert r['output'] == dev_r['SQL']
        assert r['question'] == dev_r['question'] + ' Hint: ' + dev_r['evidence']
        gold = r['output']
        pred = get_dict_value(r, field).replace('\n', ' ').replace('\t', ' ').strip()
        db = r['db']
        if use_ext:
            db += '_ext'
        res[str(i)] = pred + '\t----- bird -----\t' + db
        fg.write(gold + '\t' + db + '\n')
    json.dump(res, fp, ensure_ascii=False, indent=4)

with open(OUTPUT_DEV, 'w', encoding='utf-8') as f:
    json.dump(out_dev_data, f, ensure_ascii=False, indent=4)