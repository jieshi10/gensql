FILE=qwen2-7b-chat-spider-full_db-fs8-dev

python3 benchmark/test-suite-sql-eval/evaluation_2.py \
    --gold output/$FILE-gold.txt \
    --pred output/$FILE-pred.txt \
    --etype all \
    --db  benchmark/spider/database \
    --out_file output/$FILE-failed.json \
    --table benchmark/spider/tables.json \
    --progress_bar_for_each_datapoint \
