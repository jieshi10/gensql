import json
import glob
import os
import csv
from sql_utils_2 import dump_db_json_schema

db_name_path_map = {}
db_merged = {}

for db_path in glob.glob('benchmark/BIRD/train/train_databases/*'):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    assert db_name not in db_name_path_map
    db_name_path_map[db_name] = db_path
    assert db_name not in db_merged
    db_merged[db_name] = {}
    for o_name in [db_name]:
        schemas = dump_db_json_schema(f'{db_name_path_map[o_name]}/{o_name}.sqlite')
        for table in schemas:
            if table['name'] == 'sqlite_sequence':
                continue
            print(db_name, o_name, table['name'].lower())
            assert table['name'].lower() not in db_merged[db_name]
            db_merged[db_name][table['name'].lower()] = o_name

db_column_map = {}

def clean_key(k: str):
    if k.startswith('\ufeff'):
        return k[1:]
    return k

def process(db_path):
    db_name = db_path.split('/')[-1]
    assert db_name not in db_column_map
    db_column_map[db_name] = {}
    # print(db_path, db_name)
    assert os.path.exists(f'{db_path}/database_description/')
    for tbl_csv_path in glob.glob(f'{db_path}/database_description/*.csv'):
        tbl_csv_name = tbl_csv_path.split('/')[-1][:-4].lower()
        assert tbl_csv_name not in db_column_map[db_name]
        db_column_map[db_name][tbl_csv_name] = {}
        # print('csv:', tbl_csv_path, tbl_csv_name)
        rows = []
        with open(tbl_csv_path, encoding='utf-8', errors='ignore') as f:
            for row in csv.reader(f):
                # print(' row:', row)
                rows.append(row)
        key_map = {
            0: 'original_column_name',
            2: 'column_description',
            4: 'value_description',
        }
        if rows[0][2] == 'column_desription':
            rows[0][2] = 'column_description'
        offset = 0
        for i, k in key_map.items():
            if clean_key(rows[0][i]) != k:
                # print('error:', rows)
                offset += 1
                break
        for i, k in key_map.items():
            if i+offset<len(rows[0]):
                if clean_key(rows[0][i+offset]) != k:
                    # print('error2:', rows)
                    offset = -1
                    break
            else:
                offset = -1
                break
        if offset == -1:
            print('error:', rows)
        assert offset != -1
        for i, k in key_map.items():
            assert clean_key(rows[0][i+offset]) == k, str(rows[0])
        for row in rows[1:]:
            empty = True
            for c in row:
                if c != '':
                    empty = False
                    break
            if empty:
                continue
            column_name = None
            column_description = None
            value_description = None
            for i, k in key_map.items():
                if i+offset < len(row):
                    if k == 'original_column_name':
                        column_name = row[i+offset]
                    elif k == 'column_description':
                        column_description = row[i+offset]
                    elif k == 'value_description':
                        value_description = row[i+offset]
                    else:
                        raise ValueError
                else:
                    print('error:', rows[0], row)
            assert column_name is not None \
                and column_description is not None
            column_name = column_name.lower()
            assert column_name not in db_column_map[db_name][tbl_csv_name], \
                str(db_column_map[db_name][tbl_csv_name]) + ' ' + str(row)
            db_column_map[db_name][tbl_csv_name][column_name] = \
                column_description + (
                    f' ({value_description})'
                    if value_description is not None and value_description != ''
                    else '')

for db_path in glob.glob('benchmark/BIRD/train/train_databases/*'):
    if db_path.endswith('.json'):
        continue
    process(db_path)

res = {}

for db_path in glob.glob('benchmark/BIRD/train/train_databases/*'):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    schemas = dump_db_json_schema(f'benchmark/BIRD/train/train_databases/{db_name}/{db_name}.sqlite')
    if db_name not in db_column_map:
        print('missing db:', db_name)
        continue
    for table in schemas:
        if table['name'] == 'sqlite_sequence':
            continue
        o_name = db_merged[db_name][table['name'].lower()]
        if table['name'].lower() not in db_column_map[o_name]:
            print('missing table:', db_name, table['name'].lower(), o_name)
            continue
        for c in table['columns']:
            if c[0].lower() not in db_column_map[o_name][table['name'].lower()]:
                print('missing column:', db_name, table['name'].lower(), o_name, c)
            else:
                if db_name not in res:
                    res[db_name] = {}
                if table['name'].lower() not in res[db_name]:
                    res[db_name][table['name'].lower()] = {}
                assert c[0].lower() not in res[db_name][table['name'].lower()]
                res[db_name][table['name'].lower()][c[0].lower()] = \
                    db_column_map[o_name][table['name'].lower()][c[0].lower()].replace('\n', '; ')

with open('bird_code/descriptions_train.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

print('OK')