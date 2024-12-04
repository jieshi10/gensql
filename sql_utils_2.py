import json
import sqlite3

import process_sql


# this is very robust
def dump_db_json_schema(db_file):
    '''read table and column info'''

    conn = sqlite3.connect(db_file)
    conn.execute('pragma foreign_keys=ON')
    cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    
    tables = []

    for i, (table_name, sql) in enumerate(cursor.fetchall()):
        foreign_keys = []
        fks = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        #print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder = [[(fk[0], fk[1]), (fk[3], ), (fk[2], fk[4])] for fk in fks]
        fk_grouped = {}
        for (fk_id, sub_id), (src_fk, ), (tgt_tbl, tgt_col) in fk_holder:
            if fk_id not in fk_grouped:
                fk_grouped[fk_id] = {}
            assert sub_id not in fk_grouped[fk_id]
            fk_grouped[fk_id][sub_id] = [(src_fk, ), (tgt_tbl, tgt_col)]
        for fk_id in range(len(fk_grouped)):
            fk = [fk_grouped[fk_id][sub_id] for sub_id in range(len(fk_grouped[fk_id]))]
            assert all([tgt_tbl == fk[0][1][0] for (src_fk, ), (tgt_tbl, tgt_col) in fk])
            foreign_keys.append({
                'fk': [src_fk for (src_fk, ), (tgt_tbl, tgt_col) in fk],
                'ref': fk[0][1][0],
                'ref_key': [tgt_col for (src_fk, ), (tgt_tbl, tgt_col) in fk],
            })
        columns = []
        primary_keys = {}
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            columns.append((col[1], col[2]))
            if col[5]:
                assert col[5] not in primary_keys
                primary_keys[col[5]] = col[1]
        tables.append({
            'name': table_name,
            'sql': sql,
            'columns': columns,
            'primary_keys': [primary_keys[pk_id+1] for pk_id in range(len(primary_keys))],
            'foreign_keys': list(reversed((foreign_keys))),
        })

    return tables


IDENTIFIER = '"'


def set_identifier(i: str):
    global IDENTIFIER
    IDENTIFIER = i


# almost robust
# without considering special chars in names
def build_sql(table_name, columns, primary_keys, foreign_keys, descriptions=None):
    if descriptions is None:
        descriptions = {}
    body = ',\n'.join([f'{IDENTIFIER}{n}{IDENTIFIER} {t}' + (f' /*{descriptions[n.lower()]}*/' if n.lower() in descriptions and descriptions[n.lower()] is not None else '') for n, t in columns])
    if len(primary_keys) > 0:
        body += ',\n'
        body += f"PRIMARY KEY ({', '.join([f'{IDENTIFIER}{pk}{IDENTIFIER}' for pk in primary_keys])})"
    if len(foreign_keys) > 0:
        body += ',\n'
        body += ',\n'.join([f"FOREIGN KEY ({', '.join([f'{IDENTIFIER}{k}{IDENTIFIER}' for k in fk['fk']])}) REFERENCES {IDENTIFIER}{fk['ref']}{IDENTIFIER} ({', '.join([f'{IDENTIFIER}{k}{IDENTIFIER}' for k in fk['ref_key']])})" for fk in foreign_keys])
    return f'CREATE TABLE {IDENTIFIER}{table_name}{IDENTIFIER} (\n{body}\n);'


# not robust at all
def extract_tables_from_sql(schema, sql):
    res = set()

    def dfs(node):
        if isinstance(node, dict):
            for k, v in node.items():
                dfs(v)
            return
        if isinstance(node, list) or isinstance(node, tuple):
            for v in node:
                dfs(v)
            return
        if isinstance(node, str):
            # __xxx__
            if node.startswith('__') and node.endswith('__') and '.' not in node and node[2:-2] in schema.idMap:
                res.add(node[2:-2])
        return
    
    dfs(sql)

    return res


# not robust at all
def extract_columns_from_sql(schema, sql):
    res2 = {t: set() for t in extract_tables_from_sql(schema, sql)}

    res = set()

    def dfs(node):
        if isinstance(node, dict):
            for k, v in node.items():
                dfs(v)
            return
        if isinstance(node, list) or isinstance(node, tuple):
            for v in node:
                dfs(v)
            return
        if isinstance(node, str):
            # __xxx.yyy__
            if node.startswith('__') and node.endswith('__') and '.' in node and node[2:-2] in schema.idMap:
                t, c = tuple(node[2:-2].split('.'))
                res.add((t, c))
        return
    
    dfs(sql)

    for t, c in res:
        res2[t].add(c)
    
    return {t: list(cs) for t, cs in res2.items()}


if __name__ == '__main__':
    from pprint import pprint
    import glob
    import json
    with open('bird_code/descriptions.json', encoding='utf-8') as f:
        descriptions = json.load(f)
    for db in glob.glob('bird_code/bird_ext/*'):
        db = db.split('/')[-1]
        print(db)
        db_desc = descriptions[db.lower()]
        db = f'bird_code/bird_ext/{db}/{db}.sqlite'
        schemas = dump_db_json_schema(db)
        found = False
        for s in schemas:
            # if s['name'] == 'connected':
            for fk in s['foreign_keys']:
                if len(fk['fk']) > 1 and fk['fk'] != fk['ref_key']:
                    print(fk['fk'])
                    found = True
                    break
            if found:
                break
        if found:
            # s = schemas[0]
            pprint(s)
            print(build_sql(
                s['name'],
                s['columns'],
                s['primary_keys'],
                s['foreign_keys'],
                db_desc[s['name'].lower()]
            ))
            break

    # sql = "SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T3.born_state  =  'Alabama'"
    # sql = "SELECT COUNT(*) FROM department"
    # print(sql)
    # schema = process_sql.Schema(process_sql.get_schema(db))
    # sql = process_sql.get_sql(schema, sql)
    # # print(schema.idMap)
    # # pprint(sql)
    # print(extract_columns_from_sql(schema, sql))