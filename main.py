from typing import List, Dict, Tuple, Optional, Any

from sentence_transformers import SentenceTransformer, util
import torch
import glob
import json
from tqdm import tqdm

import numpy as np

import openai

from threading import Lock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor

from schema_free_utils_2 import get_tables_and_columns
from sql_utils_2 import build_sql, dump_db_json_schema, set_identifier
from sqlglot import parse_one, exp

import sqlglot

import nltk
import re

import sqlite3

import time

import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ZERO_SHOT_CHAT = 'zero_shot_chat'
FEW_SHOT_COMPLETION = 'few_shot_completion'

SQL_MODE = 'SQL_MODE'


###########################################################
# CONFIG

# USE_EXT = True
USE_EXT = False

# MAX_VALUES = 0
MAX_VALUES = 1

FEW_SHOT_K = 8
# FEW_SHOT_K = 4
# FEW_SHOT_K = 1
# FEW_SHOT_K = 0

# MODE = ZERO_SHOT_CHAT
MODE = FEW_SHOT_COMPLETION

MAX_SCHEMAS = 30
MAX_COLUMNS = 200

SCHEMA_SCALE = 4
COLUMN_SCALE = 6

MAX_ITER = 5

# METHOD = 'full_db'
# METHOD = 'q_table'
METHOD = 'gtrr_column'

GEN_MODE: str=SQL_MODE

USE_QUESTION_SKELETON = True
USE_SQL_SKELETON = True
ENABLE_SQL_POST_PROCESS = True

MAX_SCHEMA_LENGTH = None  # no truncation
# MAX_SCHEMA_LENGTH = (8192 - 1024) * 3  # truncate schema, measured in char

DB_NAME: str='bird'
# DB_NAME: str='spider'

MAN_FILE_SUFFIX: str='dev'

if DB_NAME == 'bird':
    # bird
    DB_SET_PATH: str='benchmark/BIRD/dev/dev_databases/*'
    # DB_SET_PATH: str='bird_code/bird_ext/*'
    TRAINING_SET_PATHS: List[str]=['benchmark/BIRD/train/train.json']
    DB_DESCRIPTION_PATH: Optional[str]=None
    # DB_DESCRIPTION_PATH: Optional[str]='bird_code/descriptions_dev.json'
    DEV_SET_PATH: str='benchmark/BIRD/dev/dev.json'
    # set_identifier('"')
    set_identifier('`')
elif DB_NAME == 'spider':
    # spider
    DB_SET_PATH: str='benchmark/spider/database/*'
    # DB_SET_PATH: str='spider_code/spider_ext/*'
    TRAINING_SET_PATHS: List[str]=['benchmark/spider/train_others.json', 'benchmark/spider/train_spider.json']
    DB_DESCRIPTION_PATH: Optional[str]=None
    DEV_SET_PATH: str='benchmark/spider/dev.json'
    # set_identifier('"')
    set_identifier('')
else:
    raise ValueError(f'Unrecognized DB_NAME: {DB_NAME}')

###########################################################


def get_file_suffix():
    if MODE == ZERO_SHOT_CHAT:
        suf = 'zs'
    elif MODE == FEW_SHOT_COMPLETION:
        suf = f'fs{FEW_SHOT_K}'
    else:
        raise ValueError(f'invalid MODE: {MODE}')
    pre = f'{METHOD}'
    if METHOD in ['gtrr_column']:
        pre += f'-S{MAX_SCHEMAS}C{MAX_COLUMNS}-SS{SCHEMA_SCALE}CS{COLUMN_SCALE}-L{MAX_ITER}'
    elif METHOD in ['q_table']:
        pre += f'-S{MAX_SCHEMAS}'
    if USE_EXT:
        pre = f'ext-{pre}'
    if USE_QUESTION_SKELETON:
        pre += '-q_skltn'
    if USE_SQL_SKELETON:
        pre += '-sql_skltn'
    if ENABLE_SQL_POST_PROCESS:
        pre += '-sql_post_process'
    if MAX_SCHEMA_LENGTH is not None:
        pre += f'-max{MAX_SCHEMA_LENGTH}'
    suffix = f'{pre}-{suf}'
    if MAN_FILE_SUFFIX is not None and MAN_FILE_SUFFIX != '':
        suffix += f'-{MAN_FILE_SUFFIX}'
    return suffix


def get_database_name(db: str):
    if USE_EXT:
        return f'{db}_ext'
    else:
        return db


###########################################################


def count_select(sql: str):
    return len(list(parse_one(sql).find_all(exp.Select)))


# t123
def possibly_temp_table(name: str):
    name = name.lower()
    return name.startswith('t') and name[1:].isnumeric()


def infer_schemas_based_on_sql(sql: str, schema: list=None) -> Tuple[List[str], Dict[str, List[str]]]:
    referenced_columns = get_tables_and_columns(sql, schemas=schema)
    referenced_columns = {
        tbl_name: columns
        for tbl_name, columns in referenced_columns.items()
        if not possibly_temp_table(tbl_name)
    }
    res = []
    for tbl_name, columns in referenced_columns.items():
        # res.append(build_sql(tbl_name, [(c, 'TEXT') for c in columns], [], []))
        res.append(tbl_name + '(' + ','.join([c for c in columns]) + ')')
    return res, referenced_columns


def plain_table_schema(table: dict) -> str:
    return 'Schema: ' + table['name'] + '(' + ', '.join([c for c, _ in table['columns']]) + ')'


def format_db_value(v: Any):
    if v is None:
        return 'NULL'
    else:
        return repr(v)


def format_db_values(values: Optional[List[Any]]):
    if values is None or len(values) == 0:
        return ''
    else:
        return f"Examples: {', '.join([format_db_value(v) for v in values])}"


def format_comment(comment: Optional[str], values: Optional[List[Any]], enable_comment=True, enable_values=True):
    if not enable_comment:
        comment = None
    if not enable_values:
        values = None
    if comment is None and (values is None or len(values) == 0):
        return None
    elif comment is None:
        return format_db_values(values)
    elif values is None:
        return comment
    else:
        return f"{comment} {format_db_values(values)}"


def plain_column(table_name: str, column_name: str, column_type: str, comment: Optional[str]=None, values: Optional[List[Any]]=None) -> str:
    text = f"Table: {table_name}, Column: {column_name}"
    if comment is not None:
        text += f"\n\nDescription:\n{comment}"
        if values is not None:
            text += "\n"
    if values is not None and len(values) > 0:
        text += "\n" + format_db_values(values)
    return text


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


def timeout_after(seconds: float):
    from multiprocessing import Process, Manager

    def func_wrapper(fn):
        
        def wrapper(*args, **kwargs):

            with Manager() as mgr:
                res = mgr.dict()
            
                def f():
                    res['ret'] = fn(*args, **kwargs)
                
                p = Process(target=f)
                p.start()
                p.join(seconds)
                if p.exitcode is None:
                    p.terminate()
                    raise TimeoutError('timeout')
                else:
                    return res['ret']

        return wrapper

    return func_wrapper


@timeout_after(10)
def exec_on_db(sqlite_path: str, query: str) -> Tuple[bool, Any]:
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return True, result
    except Exception as e:
        return False, e
    finally:
        cursor.close()
        cursor.connection.close()


def check_sql_executability(generated_sql: str, db: str):
    if generated_sql.strip() == "":
        return "Error: empty string"
    try:
        # use `EXPLAIN QUERY PLAN` to avoid actually executing
        success, res = exec_on_db(db, "EXPLAIN QUERY PLAN " + generated_sql)
        if success:
            execution_error = None
        else:
            execution_error = str(res)
        return execution_error
    except Exception as e:
        return str(e)


# extract the skeleton of the input question
def extract_question_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ["NN", "NNP", "NNS", "NNPS", "CD", "SYM", "FW", "IN"]:
            output_tokens.append("_")
        elif token in ["$", "''", "(", ")", ",", "--", ".", ":"]:
            pass
        else:
            output_tokens.append(token)

    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while "_ _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ _", "_")
    while "_ , _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ , _", "_")

    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]

    return text_skeleton


def extract_sql_skeleton(sql: str):
    sql: sqlglot.Expression = parse_one(sql, dialect='mysql')
    for table in sql.find_all(exp.Table):
        table.replace(sqlglot.to_table('TABLE_PLACEHOLDER'))
    for column in sql.find_all(exp.Column):
        column.replace(sqlglot.to_column('TABLE_PLACEHOLDER.COLUMN_PLACEHOLDER'))
    for literal in sql.find_all(exp.Literal):
        literal.replace(sqlglot.to_column('TABLE_PLACEHOLDER.LITERAL_PLACEHOLDER'))
    sql: str = sql.sql()
    return sql.replace('TABLE_PLACEHOLDER.LITERAL_PLACEHOLDER', '<LITERAL>') \
        .replace('TABLE_PLACEHOLDER.COLUMN_PLACEHOLDER', '<COLUMN>') \
        .replace('TABLE_PLACEHOLDER', '<TABLE>')


def get_query_question(question: str):
    if DB_NAME == 'bird':
        if ' Hint: ' in question:
            question = question[:question.rindex(' Hint: ')]
    if USE_QUESTION_SKELETON:
        return extract_question_skeleton(question)
    return question


def get_query_sql(sql: str):
    if USE_SQL_SKELETON:
        try:
            return extract_sql_skeleton(sql)
        except:
            return sql
    return sql


class LLMClient:
    
    def __init__(self, host: str='localhost', port: int=8080):
        self.client = openai.Client(
            api_key="empty",
            base_url=f"http://{host}:{port}/v1")
        # List models API
        models = self.client.models.list()
        assert len(models.data) == 1
        # print("Models:", models)
        self.model = models.data[0].id
        print("Use Model:", self.model)
    
    def chat(self, **kwargs):
        res = self.client.chat.completions.create(
            model=self.model, **kwargs)
        return [choice.message.content for choice in res.choices], dict(res.usage)

    def complete(self, **kwargs):
        res = self.client.completions.create(
            model=self.model, **kwargs)
        return [choice.text for choice in res.choices], dict(res.usage)


class SingleDBColumnRetriever:
    
    def __init__(self, global_retriever: 'Retriever', col_schema):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder
        global_corpus_map = global_retriever.column_corpus_map

        self.corpus = [plain_column(*c) for c in col_schema]
        self.corpus_full = col_schema
        self.corpus_name = [f'"{t}"."{c}"' for t, c, _, _, _ in col_schema]

        self.corpus_global_idx = [global_corpus_map[c] for c in self.corpus]

    @property
    def corpus_embeddings(self):
        return self.global_retriever.column_corpus_embeddings[self.corpus_global_idx]

    def batch_search_schemas(self, s: List[str], top_k: int):
        with self.lock:
            query_embedding = self.embedder.encode(s, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            assert len(hits) == len(s)
            schemas = [[(
                self.corpus_name[hit['corpus_id']],
                self.corpus[hit['corpus_id']],
                self.corpus_full[hit['corpus_id']],
                hit['score'],
                hit['corpus_id'],
            ) for hit in item] for item in hits]
            return schemas


class SingleDBTableRetriever:
    
    def __init__(self, global_retriever: 'Retriever', schema, descriptions):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder
        global_corpus_map = global_retriever.table_corpus_map

        self.corpus = [plain_table_schema(t) for t in schema]
        # self.corpus_full = [t['sql'] for t in schema]
        self.corpus_full = [build_sql(t['name'], t['columns'], t['primary_keys'], t['foreign_keys'], descriptions.get(t['name'].lower(), None)) for t in schema]
        self.corpus_name = [t['name'] for t in schema]
        self.corpus_name_idx = {n.lower(): i for i, n in enumerate(self.corpus_name)}

        self.corpus_global_idx = [global_corpus_map[t] for t in self.corpus]

    @property
    def corpus_embeddings(self):
        return self.global_retriever.table_corpus_embeddings[self.corpus_global_idx]

    def batch_search_schemas(self, s: List[str], top_k: int):
        with self.lock:
            query_embedding = self.embedder.encode(s, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            assert len(hits) == len(s)
            schemas = [[(
                self.corpus_name[hit['corpus_id']],
                self.corpus[hit['corpus_id']],
                self.corpus_full[hit['corpus_id']],
                hit['score']
            ) for hit in item] for item in hits]
            return schemas


class ExampleRetriever:
    
    def __init__(self, global_retriever: 'Retriever', training_set_paths: List[str]):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder

        print('loading training examples')
        self.example = []
        self.example_corpus_by_question = []
        self.example_corpus_by_sql = []
        for filename in training_set_paths:
            with open(filename, encoding='utf-8') as f:
                print('reading:', filename)
                data = json.load(f)
                for r in data:
                    nl = r['question']
                    if 'evidence' in r:
                        nl += ' Hint: ' + r['evidence']
                    if 'query' in r:
                        sql = r['query']
                    elif 'SQL' in r:
                        sql = r['SQL']
                    else:
                        raise ValueError('no sql')
                    nl_q = get_query_question(nl)
                    sql_q = get_query_sql(sql)
                    self.example.append({'nl': nl, 'nl_q': nl_q, 'sql': sql, 'sql_q': sql_q})
                    self.example_corpus_by_question.append(nl_q)
                    self.example_corpus_by_sql.append(sql_q)
        self.example_corpus_by_question_embeddings = self.embedder.encode(self.example_corpus_by_question, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        self.example_corpus_by_sql_embeddings = self.embedder.encode(self.example_corpus_by_sql, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

    def search_examples_by_question(self, question: str, top_k: int):
        question = get_query_question(question)
        with self.lock:
            query_embedding = self.embedder.encode(question, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.example_corpus_by_question_embeddings, top_k=top_k)
            assert len(hits) == 1
            examples = [(self.example[hit['corpus_id']], hit['score']) for hit in hits[0]]
            return examples

    def search_examples_by_sql(self, sql: str, top_k: int):
        sql = get_query_sql(sql)
        with self.lock:
            query_embedding = self.embedder.encode(sql, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.example_corpus_by_sql_embeddings, top_k=top_k)
            assert len(hits) == 1
            examples = [(self.example[hit['corpus_id']], hit['score']) for hit in hits[0]]
            return examples


class Retriever:

    def __init__(self,
                 sbert_model_path: str='./pretrained/bge-large-en-v1.5',
                #  sbert_model_path: str='./pretrained/bge-base-en-v1.5',
                #  sbert_model_path: str='./pretrained/bge-small-en-v1.5',
                 db_set_path: str=DB_SET_PATH,
                 training_set_paths: List[str]=TRAINING_SET_PATHS,
                 db_description_path: Optional[str]=DB_DESCRIPTION_PATH,
        ):
        self.embedder = SentenceTransformer(sbert_model_path)
        # self.lock = Lock()
        self.lock = BoundedSemaphore(50)  # Single GPU Total vRAM (GB) // Single Thread Required vRAM (GB)

        print('loading dbs')
        self.dbs = {}  # db schema by dump_db_json_schema
        self.db_path = {}

        self.col_schema = {}
        self.column_corpus_map = {}
        self.column_corpus = []
        self.column_retriever: Dict[str, SingleDBColumnRetriever] = {}

        self.table_corpus_map = {}
        self.table_corpus = []
        self.table_retriever: Dict[str, SingleDBTableRetriever] = {}

        if db_description_path is not None:
            with open(db_description_path, encoding='utf-8') as f:
                self.descriptions = json.load(f)
        else:
            self.descriptions = {}
        
        for db_path in glob.glob(db_set_path):
            db_name = db_path.split('/')[-1]
            print(db_path, db_name)
            self.db_path[db_name] = fr'{db_path}/{db_name}.sqlite'
            schema = dump_db_json_schema(fr'{db_path}/{db_name}.sqlite')
            self.dbs[db_name] = schema
            col_schema = []
            with sqlite3.connect(fr'{db_path}/{db_name}.sqlite') as conn:
                cursor = conn.cursor()
                for table in schema:
                    for c, t in table['columns']:
                        comment = self.descriptions.get(db_name, {}).get(table['name'].lower(), {}).get(c.lower(), None)
                        cursor.execute(f'''select distinct "{c}" from "{table['name']}" limit {MAX_VALUES}''')
                        values = [v for v, in cursor.fetchall()]
                        plain_c = plain_column(table['name'], c, t, comment, values)
                        if plain_c not in self.column_corpus_map:
                            self.column_corpus_map[plain_c] = len(self.column_corpus)
                            self.column_corpus.append(plain_c)
                        col_schema.append((table['name'], c, t, comment, values))
                    
                    plain_sql = plain_table_schema(table)
                    if plain_sql not in self.table_corpus_map:
                        self.table_corpus_map[plain_sql] = len(self.table_corpus)
                        self.table_corpus.append(plain_sql)
                cursor.close()
            self.col_schema[db_name] = col_schema
            self.column_retriever[db_name] = SingleDBColumnRetriever(self, col_schema)
            self.table_retriever[db_name] = SingleDBTableRetriever(self, schema, self.descriptions.get(db_name, {}))

        self.column_corpus_embeddings = self.embedder.encode(self.column_corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        self.table_corpus_embeddings = self.embedder.encode(self.table_corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

        self.example_retriever = ExampleRetriever(self, training_set_paths)

    def search_examples(self, question: str, top_k: int):
        return self.example_retriever.search_examples_by_question(question, top_k)
    
    def search_examples_by_sql(self, sql: str, top_k: int):
        return self.example_retriever.search_examples_by_sql(sql, top_k)


def generate_sql_zero_shot_chat(client: LLMClient, question: str, schemas: List[Tuple[str, str, str, float]]=None, **generation_configs):
    if schemas is None:
        prompt = f"Write a SQL query to answer the question.\nQuestion: {question}"
    else:
        concat_schemas = '\n'.join([schema for _, _, schema, _ in schemas])
        prompt = f"Write a SQL query to answer the question.\nDatabase Schema:\n{concat_schemas}\n\nQuestion: {question}"
    # print(prompt)
    # exit()
    res, usage = client.chat(
        messages=[{
            'role': 'user',
            'content': prompt,
        }],
        **generation_configs,
    )
    return res, usage


def generate_sql_few_shot_completion(client: LLMClient, examples: List[Tuple[dict, float]], question: str, schemas: List[Tuple[str, str, str, float]]=None, strict: Optional[bool]=None, failed_sql: Optional[str]=None, failed_sql_error_message: Optional[str]=None, **generation_configs):
    prompt = "Instruction: Write a sqlite3 SQL query to answer the question."
    if strict is not None:
        if strict:
            assert schemas is not None
            prompt += " Note that you can only use tables and columns that are explicitly given in your SQL query."
        else:
            prompt += " Note that you can also include tables and columns that are not given but are necessary to answer the question in your SQL query."
    prompt += "\n"
    for e, s in examples:
        prompt += f"Question: {e['nl']}\nSQL: {e['sql']}\n\n"
    if schemas is not None:
        concat_schemas = '\n'.join([schema for _, _, schema, _ in schemas])
        if MAX_SCHEMA_LENGTH is not None:
            concat_schemas_truncated = concat_schemas[:MAX_SCHEMA_LENGTH]
            assert concat_schemas.startswith(concat_schemas_truncated)
            assert len(concat_schemas) >= len(concat_schemas_truncated)
            concat_schemas = concat_schemas_truncated
        prompt += f"Database Schema:\n{concat_schemas}\n"
    if failed_sql is not None:
        assert failed_sql_error_message is not None
        prompt += f"\nYou may answer the question by fixing the SQL that failed to execute: {failed_sql}\nError Message: {failed_sql_error_message}\n\n"
    prompt += f"Question: {question}\nSQL: SELECT "
    # print(prompt)
    # exit()
    stop = ['Question', 'Instruction', '\n\n']
    if 'stop' in generation_configs:
        generation_configs['stop'] += stop
    else:
        generation_configs['stop'] = stop
    res, usage = client.complete(
        prompt=prompt,
        **generation_configs,
    )
    return ['SELECT ' + s for s in res], usage


def prepare_sql_generation(client: LLMClient, retriever: Retriever, mode: str, question: str):
    assert mode in [ZERO_SHOT_CHAT, FEW_SHOT_COMPLETION]
    if mode == ZERO_SHOT_CHAT:
        examples = None

        def generate_sql(schemas: List[Tuple[str, str, float]]=None):
            res, usage = generate_sql_zero_shot_chat(client, question, schemas,
                max_tokens=128,
                n=1,
                temperature=0,
            )
            assert len(res) == 1
            infer_sql = res[0]
            return infer_sql, usage
        
    else:
        assert mode == FEW_SHOT_COMPLETION
        examples = retriever.search_examples(question, FEW_SHOT_K)

        def generate_sql(schemas: List[Tuple[str, str, float]]=None, previous_sql: Optional[str]=None, failed_sql: Optional[str]=None, failed_sql_error_message: Optional[str]=None):
            if previous_sql is not None:
                examples_by_sql = retriever.search_examples_by_sql(previous_sql, FEW_SHOT_K//2)
                examples_copied = sum(map(list, list(zip(examples[:FEW_SHOT_K//2], examples_by_sql))), [])
                ret_extra = [examples_copied]
            else:
                examples_copied = examples
                ret_extra = []

            res, usage = generate_sql_few_shot_completion(client, examples_copied, question, schemas,
                failed_sql=failed_sql,
                failed_sql_error_message=failed_sql_error_message,
                max_tokens=256,
                n=1,
                temperature=0,
            )
            assert len(res) == 1
            infer_sql = res[0]
            return infer_sql, usage, *ret_extra
    
    return examples, generate_sql


def convert_retrieved_columns_to_table_schemas(retriever: Retriever, r: dict, retrieved_columns, enable_comment=True, enable_values=True):
    pks = {}
    pks_orig = {}
    fks = {}
    for table in retriever.dbs[get_database_name(r['db_id'])]:
        pks[table['name'].lower()] = [pk.lower() for pk in table['primary_keys']]
        pks_orig[table['name']] = table['primary_keys']
        fks[table['name'].lower()] = [{
            'fk': [k.lower() for k in fk['fk']],
            'ref': fk['ref'].lower(),
            'ref_key': [k.lower() for k in fk['ref_key']],
            'original_fk': fk,
        } for fk in table['foreign_keys']
        if all([k is not None for k in fk['fk']])
            and all([k is not None for k in fk['ref_key']])]

    columns = {}
    for table_name, column_name, column_type, comment, values in retriever.col_schema[get_database_name(r['db_id'])]:
        if table_name.lower() not in columns:
            columns[table_name.lower()] = {}
        assert column_name.lower() not in columns[table_name.lower()]
        columns[table_name.lower()][column_name.lower()] = (
            (column_name, column_type),
            # f"{comment} Examples: {', '.join([repr(v) for v in values])}"
            format_comment(comment, values, enable_comment, enable_values)
        )

    schemas = {}
    descriptions = {}
    for _, _, (table_name, column_name, column_type, comment, values), _ in retrieved_columns:
        if table_name not in schemas:
            schemas[table_name] = []
            descriptions[table_name] = {}
        schemas[table_name].append((column_name, column_type))
        if column_name is not None:
            descriptions[table_name][column_name.lower()] = format_comment(comment, values, enable_comment, enable_values)  # f"{comment} Examples: {', '.join([repr(v) for v in values])}"
    
    schema_columns = {
        t.lower(): set([c.lower() for c, _ in schemas[t] if c is not None])
        for t in schemas
    }

    def add_column_to_table_if_not_exists(t, k):
        if k not in schema_columns[t.lower()]:
            c, d = columns[t.lower()][k]
            schemas[t].append(c)
            descriptions[t][k] = d
            schema_columns[t.lower()].add(k)
    
    valid_fks = {}
    for t in schemas:
        for k in pks[t.lower()]:
            add_column_to_table_if_not_exists(t, k)
        valid_fks[t] = []
        for fk in fks[t.lower()]:
            if fk['ref'] in schema_columns:
                ref_table = None
                for tn in schemas:
                    if tn.lower() == fk['ref']:
                        ref_table = tn
                        break
                assert ref_table is not None
                for k in fk['fk']:
                    add_column_to_table_if_not_exists(t, k)
                for k in fk['ref_key']:
                    add_column_to_table_if_not_exists(ref_table, k)
                valid_fks[t].append(fk['original_fk'])

    retrieved_schemas = [(t, None, build_sql(t, [cc for cc in c if cc != (None, None)], pks_orig[t], valid_fks[t], descriptions[t]), 1.0) for t, c in schemas.items()]
    
    return retrieved_schemas


# a += b
def add_usage_inplace(a: dict, b: Optional[dict]):
    assert a is not None
    if b is None:
        return
    for k in a:
        a[k] += b[k]


def process_record_full_db(client: LLMClient, retriever: Retriever, r: dict, mode: str):
    start_time = time.time()

    examples, generate_sql = prepare_sql_generation(client, retriever, mode, r['question'])

    full_columns = [
        (
            f'"{tname}"."{cname}"',
            None,
            (tname, cname, ctype, c, v),
            1.0
        )
        for tname, cname, ctype, c, v in retriever.column_retriever[get_database_name(r['db_id'])].corpus_full
    ]
    full_schemas = convert_retrieved_columns_to_table_schemas(retriever, r, full_columns)

    concat_schemas = full_schemas
    
    infer_sql, usage = generate_sql(concat_schemas)

    if ENABLE_SQL_POST_PROCESS:
        error = check_sql_executability(infer_sql, retriever.db_path[get_database_name(r['db_id'])])
        if error is not None:
            infer_sql, fixed_usage = generate_sql(concat_schemas, failed_sql=infer_sql, failed_sql_error_message=error)
            add_usage_inplace(usage, fixed_usage)

    end_time = time.time()

    return {
        'id': r['id'],
        'question': r['question'],
        'output': r['query'],
        'db': r['db_id'],
        'fs_examples': examples,
        # 'full_columns': full_columns,
        'full_schemas': full_schemas,
        'infer': infer_sql,
        'usage': usage,
        'time': end_time - start_time,
    }


def process_record_retrieve_tables_based_on_question(client: LLMClient, retriever: Retriever, r: dict, mode: str):
    start_time = time.time()
    
    examples, generate_sql = prepare_sql_generation(client, retriever, mode, r['question'])

    search_query = 'Question:' + r['question']
    retrieve_schemas_based_on_question = retriever.table_retriever[get_database_name(r['db_id'])] \
        .batch_search_schemas([search_query], MAX_SCHEMAS)[0]
    retrieve_tables = set([table[0] for table in retrieve_schemas_based_on_question])
    retrieve_columns_based_on_question = [
        (
            f'"{tname}"."{cname}"',
            None,
            (tname, cname, ctype, c, v),
            1.0
        )
        for tname, cname, ctype, c, v in retriever.column_retriever[get_database_name(r['db_id'])].corpus_full
        if tname in retrieve_tables
    ]

    retrieve_tables = retrieve_schemas_based_on_question

    retrieve_schemas_based_on_question = convert_retrieved_columns_to_table_schemas(retriever, r, retrieve_columns_based_on_question)

    concat_schemas = retrieve_schemas_based_on_question

    infer_sql, usage = generate_sql(concat_schemas)

    if ENABLE_SQL_POST_PROCESS:
        error = check_sql_executability(infer_sql, retriever.db_path[get_database_name(r['db_id'])])
        if error is not None:
            infer_sql, fixed_usage = generate_sql(concat_schemas, failed_sql=infer_sql, failed_sql_error_message=error)
            add_usage_inplace(usage, fixed_usage)
        # infer_sql = sql_post_process(infer_sql, retriever.dbs[get_database_name(r['db_id'])])

    end_time = time.time()

    return {
        'id': r['id'],
        'question': r['question'],
        'output': r['query'],
        'db': r['db_id'],
        'fs_examples': examples,
        'search_query': search_query,
        # 'retrieve_tables': retrieve_tables,
        # 'retrieve_columns': retrieve_columns_based_on_question,
        'retrieve_schemas': retrieve_schemas_based_on_question,
        'infer': infer_sql,
        'usage': usage,
        'time': end_time - start_time,
    }


def process_record_generate_then_retrieve_repeat_column(client: LLMClient, retriever: Retriever, r: dict, mode: str):
    start_time = time.time()

    examples, generate_sql = prepare_sql_generation(client, retriever, mode, r['question'])

    infer_sql, usage = generate_sql()
    previous_retrieve_columns = set()
    stop_reason = 'max_iteration'
    rounds = [{
        'infer': infer_sql,
        'usage': usage,
        'time': time.time() - start_time,
    }]
    for round_id in range(MAX_ITER):
        search_query = 'Question:' + r['question']
        try:
            schemas, referenced_columns = infer_schemas_based_on_sql(infer_sql, schema=(retriever.dbs[get_database_name(r['db_id'])] if round_id > 0 else None))
            n = len(schemas)
            if n == 0:
                raise ValueError('empty schemas')
            search_query += '\nSchema:' + ';'.join(schemas)
        except Exception as e:
            print(e)
            schemas = None
            n = MAX_SCHEMAS
            referenced_columns = None
        
        retrieve_all_columns_based_on_infer_schema = retriever.column_retriever[get_database_name(r['db_id'])] \
            .batch_search_schemas([search_query], len(retriever.column_retriever[get_database_name(r['db_id'])].corpus))[0]

        if schemas is not None:
            n = min(n*SCHEMA_SCALE, MAX_SCHEMAS)
            retrieve_schemas_based_on_infer_schema = retriever.table_retriever[get_database_name(r['db_id'])] \
                .batch_search_schemas([search_query], n)[0]
            table_columns_dict = {
                table_name: []
                for table_name, _, _, _ in retrieve_schemas_based_on_infer_schema
            }
            max_columns = 1
            for _, columns in referenced_columns.items():
                max_columns = max(max_columns, len(columns))
            referenced_columns_cnt = {
                table_name: max(len(referenced_columns[table_name.lower()]), 1)
                    if table_name.lower() in referenced_columns else max_columns
                for table_name in table_columns_dict
            }
            for column in retrieve_all_columns_based_on_infer_schema:
                table_name = column[2][0]
                if table_name in table_columns_dict \
                        and len(table_columns_dict[table_name]) < min(
                            referenced_columns_cnt[table_name]*COLUMN_SCALE, MAX_COLUMNS):
                    table_columns_dict[table_name].append(column)
            retrieve_tables = retrieve_schemas_based_on_infer_schema
            retrieve_columns_based_on_infer_schema = sum([columns for _, columns in table_columns_dict.items()], [])
        else:
            retrieve_tables = None
            retrieve_columns_based_on_infer_schema = retrieve_all_columns_based_on_infer_schema[:MAX_COLUMNS]

        retrieve_columns_based_on_infer_schema.sort(key=lambda column: column[-1])
        retrieve_columns_based_on_infer_schema = [column[:-1] for column in retrieve_columns_based_on_infer_schema]

        retrieve_schemas_based_on_infer_schema = convert_retrieved_columns_to_table_schemas(retriever, r, retrieve_columns_based_on_infer_schema)
        
        current_retrieve_columns = set([name.lower() for name, _, _, _ in retrieve_columns_based_on_infer_schema])
        if current_retrieve_columns == previous_retrieve_columns:
            stop_reason = 'early_stopping'
            break
        concat_schemas = retrieve_schemas_based_on_infer_schema
        infer_sql, usage, current_examples = generate_sql(concat_schemas, infer_sql)
        
        if ENABLE_SQL_POST_PROCESS:
            error = check_sql_executability(infer_sql, retriever.db_path[get_database_name(r['db_id'])])
            if error is not None:
                infer_sql, fixed_usage, current_examples = generate_sql(concat_schemas, infer_sql, failed_sql=infer_sql, failed_sql_error_message=error)
                add_usage_inplace(usage, fixed_usage)
        
        rounds.append({
            'fs_examples': current_examples,
            'infer_schemas': schemas,
            'search_query': search_query,
            # 'retrieve_tables': retrieve_tables,
            # 'retrieve_columns': retrieve_columns_based_on_infer_schema,
            'retrieve_schemas': retrieve_schemas_based_on_infer_schema,
            'infer': infer_sql,
            'usage': usage,
            'time': time.time() - start_time,
        })
        previous_retrieve_columns = current_retrieve_columns

    return {
        'id': r['id'],
        'question': r['question'],
        'output': r['query'],
        'db': r['db_id'],
        'fs_examples': examples,
        'rounds': rounds,
        'stop_reason': stop_reason,
        'infer': infer_sql,
        'time': time.time() - start_time,
    }


def process_record(client: LLMClient, retriever: Retriever, r: dict):
    if GEN_MODE == SQL_MODE:
        if METHOD == 'full_db':
            func = process_record_full_db
        elif METHOD == 'q_table':
            func = process_record_retrieve_tables_based_on_question
        elif METHOD == 'gtrr_column':
            func = process_record_generate_then_retrieve_repeat_column
        else:
            raise ValueError(f'invalid METHOD: {METHOD}')
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')
    
    # return func(client, retriever, r, MODE)
    try:
        return func(client, retriever, r, MODE)
    except Exception as e:
        print('error:', e)
        return None


if __name__ == '__main__':
    client = LLMClient(host='127.0.0.1', port=8080)
    retriever = Retriever()

    if GEN_MODE == SQL_MODE:
        with open(DEV_SET_PATH, encoding='utf-8') as f:
            data = json.load(f)
            for i, r in enumerate(data):
                r['id'] = i
                if DB_NAME == 'bird':
                    r['original_question'] = r['question']
                    r['question'] += ' Hint: ' + r['evidence']
                    r['query'] = r['SQL']
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')

    # data = data[:20]

    start = time.time()

    with ThreadPoolExecutor(max_workers=50) as executor:
    # with ThreadPoolExecutor(max_workers=1) as executor:
        res_data = list(tqdm(executor.map(lambda r: process_record(client, retriever, r), data), total=len(data)))

    end = time.time()
    print('elapsed:', end - start, 'seconds')

    assert len(data) == len(res_data)
    for i, o in zip(data, res_data):
        if o is None:
            continue
        assert i['id'] == o['id']
        assert i['question'] == o['question']
        assert i['query'] == o['output']
        if DB_NAME == 'bird':
            assert i['SQL'] == o['output']
        assert i['db_id'] == o['db']
        o.pop('id')

    if GEN_MODE == SQL_MODE:
        out_file = f'output/{client.model}-{DB_NAME}-{get_file_suffix()}.json'
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res_data, f, ensure_ascii=False, indent=4)
    print('saved:', out_file)