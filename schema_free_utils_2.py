import sqlglot
import sqlglot.errors
from sqlglot.optimizer.scope import traverse_scope

from typing import List


def get_tables_and_columns(sql: str, schemas: List[dict]=None):
    try:
        sql = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        sql = sqlglot.parse_one(sql, dialect='mysql')
    table_with_alias = {}
    table_columns = {}  # referenced_columns
    for table in sql.find_all(sqlglot.exp.Table):
        if table.name.lower() not in table_columns:
            table_columns[table.name.lower()] = []
        if table.alias != '':
            table_with_alias[table.alias.lower()] = table.name.lower()
        else:
            table_with_alias[table.name.lower()] = table.name.lower()
    for scope in traverse_scope(sql):
        unqualified_columns = []
        for column in scope.columns:
            if column.table != '':
                if column.table.lower() in table_with_alias:
                    tbl_name = table_with_alias[column.table.lower()]
                    if column.name.lower() not in table_columns[tbl_name]:
                        table_columns[tbl_name].append(column.name.lower())
                else:
                    table_columns[column.table.lower()] = [column.name.lower()]
            else:
                if column.name.lower() not in unqualified_columns:
                    unqualified_columns.append(column.name.lower())
        for column in unqualified_columns:
            for table in scope.tables:
                if column not in table_columns[table.name.lower()]:
                    table_columns[table.name.lower()].append(column)
    
    if schemas is not None:
        schemas = {
            table['name'].lower(): set([column.lower() for column, _ in table['columns']])
            for table in schemas
        }
        table_columns = {
            table: [column for column in columns if column in schemas[table]]
            for table, columns in table_columns.items() if table in schemas
        }
    
    return table_columns
            

if __name__ == '__main__':
    from bird_code.sql_utils_2 import dump_db_json_schema

    schemas = dump_db_json_schema('benchmark/BIRD/dev/dev_databases/california_schools/california_schools.sqlite')
    # for s in schemas:
    #     print(s['sql'])
    print(get_tables_and_columns('SELECT DISTINCT T1.AdmFName1, T1.District FROM schools AS T1 INNER JOIN ( SELECT admfname1 FROM schools GROUP BY admfname1 ORDER BY COUNT(admfname1) DESC LIMIT 2 ) AS T2 ON T1.AdmFName1 = T2.admfname1'))
    print(get_tables_and_columns('SELECT DISTINCT T1.AdmFName1, T1.District FROM schools AS T1 INNER JOIN ( SELECT admfname1 FROM schools GROUP BY admfname1 ORDER BY COUNT(admfname1) DESC LIMIT 2 ) AS T2 ON T1.AdmFName1 = T2.admfname1', schemas=schemas))

    schemas = dump_db_json_schema('benchmark/spider/database/concert_singer/concert_singer.sqlite')
    # for s in schemas:
    #     print(s['sql'])
    print(get_tables_and_columns('select CONCERT_NAME from concert where stadium_id = (select `stadium_id` from `STADIUM` order by CAPACITY desc limit 1);'))
    print(get_tables_and_columns('select CONCERT_NAME from concert where stadium_id = (select `stadium_id` from `STADIUM` order by CAPACITY desc limit 1);', schemas=schemas))

    # print(get_tables_and_columns('SELECT count(* FROM singer'))
    print(get_tables_and_columns('SELECT count(*) FROM `singer`'))
    print(get_tables_and_columns('SELECT count(*) FROM singer'))
    print(get_tables_and_columns('select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1'))
    print(get_tables_and_columns('SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1'))
    print(get_tables_and_columns('SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)'))
    print(get_tables_and_columns('SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014'))