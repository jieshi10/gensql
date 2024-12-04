import json

filename = r'output/llama3-70b-chat-bird-ext-gtrr_column-S30C200-SS4CS6-L5-q_skltn-sql_skltn-sql_post_process-fs8.json'

fields = [
    # 'usage.prompt_tokens',

    'rounds.0.usage.prompt_tokens',
    'rounds.1.usage.prompt_tokens',

    'rounds.2.usage.prompt_tokens',
    'rounds.3.usage.prompt_tokens',
    'rounds.4.usage.prompt_tokens',
    'rounds.5.usage.prompt_tokens',
]


def get_dict_value(r: dict, field: str, default_value):
    
    def check_key(r, k):
        if isinstance(r, list):
            if k >= len(r) or k < -len(r):  # [-len(r), len(r))
                return False
        elif isinstance(r, dict):
            if k not in r:
                return False
        else:
            raise ValueError
        return True

    if '.' in field:
        field = field.split('.')
        for k in field:
            try:
                k = int(k)
            except:
                pass
            if not check_key(r, k):
                return default_value
            r = r[k]
        return r
    else:
        if not check_key(r, field):
            return default_value
        return r[field]


with open(filename, encoding='utf-8') as f:
    data = json.load(f)

total = 0

for r in data:
    for field in fields:
        total += get_dict_value(r, field, 0)

print(total / len(data))
