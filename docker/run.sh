docker run -it --rm --gpus all --shm-size=128g \
    -v $(pwd):/mnt/data/ \
    -v $(pwd)/nltk_data/:/root/nltk_data/ \
    text-to-sql
