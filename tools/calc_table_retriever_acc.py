import json
from sql_utils_2 import dump_db_json_schema
from schema_free_utils_2 import get_tables_and_columns

# DATABASE_DIR = 'spider_code/spider_ext'
DATABASE_DIR = 'bird_code/bird_ext'
use_ext = True

filename = r'output/llama3-70b-chat-bird-ext-gtrr_column-S30C200-SS4CS6-L5-q_skltn-sql_skltn-sql_post_process-fs8.json'

# fields = ['gold_schemas']
# fields = ['full_schemas']
# fields = ['retrieve_schemas']
# fields = ['rounds.1.retrieve_schemas']
fields = ['rounds.-1.retrieve_schemas']

max_n_tables_gold_sql = 0

with open(filename, encoding='utf-8') as f:
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


n_correct, n_pred, n_gold = 0, 0, 0

n_none = 0

for r in data:
    if r is None:
        n_none += 1
        continue
    pred = []
    for field in fields:
        pred += get_dict_value(r, field)
    db = r['db']
    if use_ext:
        db += '_ext'
    schemas = dump_db_json_schema(f"{DATABASE_DIR}/{db}/{db}.sqlite")
    pred = list(map(lambda x: x[0].lower(), pred))
    gold = list(get_tables_and_columns(r['output'], schemas=schemas).keys())
    max_n_tables_gold_sql = max(max_n_tables_gold_sql, len(gold))
    pred = set(pred)
    gold = set(gold)
    n_correct += len(pred & gold)
    n_pred += len(pred)
    n_gold += len(gold)

precision = n_correct / n_pred
recall = n_correct / n_gold
f1 = (2*precision*recall)/(precision+recall)

print(filename, fields)
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)

print()

print('max_n_tables_gold_sql:', max_n_tables_gold_sql)
print('n_none:', n_none)