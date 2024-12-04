import sqlite3

import json
import os

import glob


db_dir = 'bird_code/bird_ext'

assert not os.path.exists(db_dir)
os.mkdir(db_dir)

with open('bird_code/bird_merged_schema_3.json', encoding='utf-8') as f:
    dbs = json.load(f)

db_path_map = {}
dev_dbs = set()

for db_path in glob.glob('benchmark/BIRD/dev/dev_databases/*'):
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    assert db_name not in db_path_map
    db_path_map[db_name] = db_path
    dev_dbs.add(db_name + '_ext')

for db_path in glob.glob('benchmark/BIRD/train/train_databases/*'):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    assert db_name not in db_path_map
    db_path_map[db_name] = db_path

print('=' * 30)
print('dev_dbs:', len(dev_dbs))

for db in dbs:
    if db['name'] not in dev_dbs:
        continue
    db_path = os.path.join(db_dir, db['name'])
    os.mkdir(db_path)
    db_file = os.path.join(db_path, db['name'] + '.sqlite')
    with sqlite3.connect(db_file) as conn_t:
        # conn_t.text_factory = lambda x: str(x, 'latin1')
        c_t = conn_t.cursor()
        all_tables = []
        for o_name in db['original']:
            with sqlite3.connect(f'{db_path_map[o_name]}/{o_name}.sqlite') as conn:
                # conn.text_factory = lambda x: str(x, 'latin1')
                c = conn.cursor()
                c.execute("select name, sql from sqlite_master where type='table';")
                tables = []
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
                for t in tables:
                    tbl_name = t['name']
                    sql = t['sql']

                    new_name = tbl_name
                    new_sql = sql
                    
                    # print(tbl_name)
                    # print(sql)
                    c.execute(f"select * from \"{tbl_name}\";")
                    d = c.fetchall()
                    # for r in d[:3]:
                    #     print(r)

                    c_t.execute(new_sql)
                    if len(d) > 0:
                        c_t.executemany(f"insert into \"{new_name}\" values ({', '.join(['?'] * len(d[0]))})", d)
                    conn_t.commit()

                    # print('=' * 30)
                all_tables += tables
        s_tables = set()
        for t in all_tables:
            if t in db['tables']:
                s_tables.add(tuple(sorted(t.items())))
            # print('*', t['name'])
        # print('+', list(map(lambda x: x['name'], db['tables'])))
        assert len(s_tables) == len(db['tables'])
        # exit()
