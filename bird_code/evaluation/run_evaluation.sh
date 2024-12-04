db_root_path='benchmark/BIRD/dev/dev_databases/'
filename=qwen2-7b-chat-bird-full_db-fs8-dev
diff_json_path="output/$filename-dev.json"
predicted_sql_path="output/$filename-pred.json"
ground_truth_path="output/$filename-gold.sql"
out_json_path="output/$filename-out.json"
num_cpus=16
meta_time_out=30.0

echo '''starting to compare with knowledge for ex'''
python3 -u bird_code/evaluation/evaluation2.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --meta_time_out ${meta_time_out} \
--diff_json_path ${diff_json_path} --out_json_path ${out_json_path}

# echo '''starting to compare with knowledge for ves'''
# python3 -u bird_code/evaluation/evaluation_ves2.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --meta_time_out ${meta_time_out} \
# --diff_json_path ${diff_json_path}
