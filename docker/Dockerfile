FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install -y python3 python3-dev python3-pip python3-venv
RUN apt install -y git

# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /root

COPY requirements.txt .
RUN pip3 install packaging ninja
RUN pip3 install -r requirements.txt

RUN pip3 install openai scikit-learn

RUN pip3 install SQLAlchemy
