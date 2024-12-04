# Gen-SQL: Efficient Text-to-SQL By Bridging Natural Language Question And Database Schema With Pseudo-Schema

## Prerequisite

### Environment
- Python 3.10
- CUDA 12.1

*Refer to [requirements.txt](requirements.txt) for required Python packages.*

- NLTK: Run the following code in Python interpreter to download nltk data.
  ```python
  >>> import nltk
  >>> nltk.download('punkt')
  >>> nltk.download('averaged_perceptron_tagger')
  ```

### Pre-trained Models

**Retriever Model:**
- [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

Put the retriever model in [pretrained](pretrained).

*LLMs are omitted.*

### Datasets Preprocessing

#### Spider
- Download [Spider](https://yale-lily.github.io/spider) dataset, and unzip `spider.zip` in the [benchmark](benchmark) directory.

#### BIRD
- Download [BIRD](https://bird-bench.github.io/) dataset, and unzip `train.zip` and `dev.zip` in the [benchmark/BIRD](benchmark/BIRD) directory.

*If you are NOT interested in the mass datasets, you may safely skip the next steps.*

#### Spider to Spider-mass
- Merge databases in the root directory (of this project):
  ```shell
  python spider_code/merge_spider_db.py
  ```
  After a few minutes (less than 10 minutes on my server), you can find the merged databases in the directory named `spider_code/spider_ext`.

#### BIRD to BIRD-mass
- Merge databases in the root directory (of this project):
  ```shell
  python bird_code/merge_bird_dev_db.py
  ```
  It may take about an hour until you can find the merged databases in the directory named `bird_code/bird_ext`.

## Running Experiments

- Start openai-compatible vllm server:
  ```shell
  python -m vllm.entrypoints.openai.api_server --model [YOUR MODEL]
  ```
  Here is the [link](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for quick reference.
- Run code:
  ```shell
  python main.py
  ```
  The results will be saved to [output](output).
- Execution accuracy:
  - Convert output:
    
    **Spider or Spider-mass:**
    ```shell
    python spider_code/convert_output_ext.py
    ```
    **BIRD or BIRD-mass:**
    ```shell
    python bird_code/convert_output_ext.py
    ```
  - Run evaluation:
    
    **Spider or Spider-mass:**
    ```shell
    . spider_code/eval-sql.sh
    ```
    **BIRD or BIRD-mass:**
    ```shell
    . bird_code/evaluation/run_evaluation.sh
    ```