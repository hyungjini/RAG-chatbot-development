# [00] install
!pip install datasets
!pip install requests
!pip install tiktoken
!pip install numpy
!pip install openai --upgrade
!pip install nltk
import nltk
nltk.download('punkt')
!pip install langchain
!pip install langchain-openai
!pip install tiktoken
!pip install sentence-transformers

# [02] import
import os
import getpass
import openai
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = 'your_openaikey'
client = OpenAI()
import nltk
import json
from nltk.tokenize import word_tokenize
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import base64

# [03] Fine-tunning전, TOKEN 수 계산
def count_tokens_in_jsonl(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    total_tokens = 0
    for line in lines:
        data = json.loads(line)
        tokens = word_tokenize(str(data))  # Tokenize the data
        total_tokens += len(tokens)
    return total_tokens
total_tokens_num = count_tokens_in_jsonl(jsonl_file_path)
## 사용 예시
# file_path = '/content/sample_data/Traning_dataset_final.jsonl'
# print(f'Total tokens: {total_tokens_num}')

# [04] Fine-tunning전, 가격 계산

# [04] 토큰 수, 토큰 당 비용, 에폭 수를 기반으로 총 가격을 계산하는 함수
def price_of_tokens(total_tokens_num, cost_per_token, epochs):
    # 입력 변수 타입 확인 및 변환
    if isinstance(total_tokens_num, str):
        try:
            total_tokens_num = float(total_tokens_num)
        except ValueError:
            raise ValueError("total_tokens_num은 숫자여야 합니다.")
    if isinstance(cost_per_token, str):
        try:
            cost_per_token = float(cost_per_token)
        except ValueError:
            raise ValueError("cost_per_token은 숫자여야 합니다.")
    if isinstance(epochs, str):
        try:
            epochs = float(epochs)
        except ValueError:
            raise ValueError("epochs는 숫자여야 합니다.")
    # 총 토큰 수 계산 및 총 가격 계산
    total_tokens = int(total_tokens_num / 1000)
    total_price = total_tokens * cost_per_token * epochs
    return total_price

## 토큰 당 비용 (달러 단위)
# cost_per_token = 0.008
## 에폭 수
# epochs = input('에폭수치를 입력해주세요 : ')
## 파인튜닝 총 비용 계산
# total_cost = price_of_tokens(total_tokens_num, cost_per_token, epochs)
## 파인튜닝 비용 출력
# print(f'총 가격 = (토큰 당 기본 비용) x (토큰수) x (에폭 수치)')
# print(f'총가격 = {cost_per_token} X {total_tokens_num}/1000 X {epochs} = ${total_cost:.2f}')

#[05-1] 학습 파일에 대한 몇 가지 임시 검사
# Run preliminary checks
import json
# Load the training set
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]
# Training dataset stats
print("Number of examples in training set:", len(training_dataset))
print("First example in training set:")
for message in training_dataset[21506]["messages"]:
    print(message)

# [05-2] 토큰 수의 유효성을 검사(개별 예제는 gpt-35-turbo-0613 모델의 입력 토큰 제한인 4096개의 토큰 이하로 유지해야 함.) + 데이터 셋 대략적인 평가
import json
import tiktoken
import numpy as np
from collections import defaultdict
# gpt-4, turbo, text-embedding-ada-002 모델에서 사용되는 기본 인코딩
encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1): #메시지에서의 토큰 수 계산;입력으로 받은 메시지 목록에서 각 메시지와 메시지의 'name' 필드를 인코딩하여 총 토큰 수를 계산
    num_tokens = 0
    for message in messages:
        # 메시지 당 기본 토큰 수
        num_tokens += tokens_per_message
        for key, value in message.items():
            # 메시지의 각 항목을 인코딩하여 토큰 수 계산
            num_tokens += len(encoding.encode(value))
            if key == "name":
                # 이름에 대한 추가 토큰 수
                num_tokens += tokens_per_name
    # 메시지 처리 완료 후 기본 토큰 추가
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages): #조수 역할 메시지의 토큰 수 계산;입력으로 받은 메시지 목록 중 'role'이 'assistant'인 메시지의 내용을 인코딩하여 토큰 수를 계산
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            # assistant의 역할을 하는 메시지 내용의 토큰 수 계산
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name): # 값의 분포를 출력하는 함수; 데이터의 분포를 이해하고 데이터셋의 특성을 파악하기 위해 사용됨. 이를 통해 데이터의 중심 경향성, 변동성, 왜도 등을 간략하게 확인할 수 있음.
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
files = ['/content/sample_data/Traning_dataset_final.jsonl']

for file in files:
    # 파일 처리 시작
    print(f"Processing file: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        # 파일 내 각 줄을 JSON 객체로 변환
        dataset = [json.loads(line) for line in f]

    total_tokens = []
    assistant_tokens = []

    for ex in dataset:
        # 메시지 추출
        messages = ex.get("messages", {})
        # 총 토큰 수와 assistant 토큰 수 계산
        total_tokens.append(num_tokens_from_messages(messages))
        assistant_tokens.append(num_assistant_tokens_from_messages(messages))

    # 토큰 수의 분포 출력
    print_distribution(total_tokens, "total tokens")
    print_distribution(assistant_tokens, "assistant tokens")
    # 구분선
    print('*' * 50)
    # 해석 방법에 대한 설명
    print("해석 방법\n")
    print("1. 최소값(min)과 최대값(max): 데이터셋의 토큰 수 범위를 파악함. 이는 데이터의 전반적인 크기와 구조를 이해할 수 있음.")
    print("2. 평균(mean)과 중앙값(median): 데이터의 중심 경향성에 대한 수치. 평균은 데이터의 총합을 개수로 나눈 값이고, 중앙값은 데이터를 정렬했을 때 중앙에 위치하는 값임.")
    print("3. 5% 백분위수(p5)와 95% 백분위수(p95): 대부분의 데이터가 이 범위 내에 분포함을 보여줍니다. 데이터의 극단적인 값과 이상치의 존재 여부를 파악할 수 있음.")
    print("4. 데이터 분포의 대칭성과 일관성: 평균과 중앙값이 근접할수록 데이터는 대칭적이고. 또한, 분포의 일관성을 통해 데이터의 예측 가능성을 검토할 수 있다?.")
    print("\n각 분포를 통해 데이터셋의 특성과 품질을 평가하고, 필요한 데이터 처리와 분석 방향을 결정할 수 있다.... 해야할까..")
