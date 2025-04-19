import re
import unicodedata
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

def unicodeToAscii(s):
    # 한글 문자 찾기
    s = re.sub(r'(?i)cc-by.*$', '', s)
    hangul_pattern = re.compile('[가-힣ㄱ-ㅎㅏ-ㅣ]')
    result = []
    for c in s:
        if hangul_pattern.match(c):
            result.append(c)
        else:
            for c_ in unicodedata.normalize('NFD', c): #결합문자 분해
                if unicodedata.category(c_) != 'Mn': #결합문자 제거
                    result.append(c_)
    return ''.join(result)

def Norm_String(s):
    s = unicodeToAscii(s.lower().strip())
    # ([.!?]): 마침표(.), 느낌표(!), 물음표(?) 중 하나를 찾고 괄호로 그룹화(캡처)합니다.
    # r" \1": 찾은 구두점 앞에 공백을 추가합니다. \1은 첫 번째 캡처 그룹(구두점)을 참조합니다.
    s = re.sub(r"([.!?])", r" \1", s)
    # [^a-zA-Z!?]+: 영문자(a-z, A-Z)와 한글 느낌표(!), 물음표(?)가 아닌 문자들을 하나 이상(+) 찾습니다.
    # 이것은 숫자, 다른 구두점, 다른 언어의 문자 등을 포함합니다.
    # r" ": 찾은 문자들을 단일 공백으로 대체합니다.
    s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ!?]+", r" ", s)
    return s.strip()

def read_language(L1, L2, reverse=False, verbose=False):
    print("read languages data...")
    # strip() : 문자제거
    # split() : 문자열 내부에 있는 공백 또는 특별한 문자를 구분해서, 리스트 아이템으로 만듦.
    pairs = []
    Encode_lang = []
    Decode_lang = []
    pf = open('%s2%s.txt' % (L1, L2), encoding='utf-8').read().strip().split('\n')
    for ll in pf:
        parts = ll.split('\t')
        if len(parts) >2:
            L1_lang = Norm_String(parts[0])
            L2_lang = Norm_String(parts[1])
        if reverse:
            pairs.append([L2_lang, L1_lang])
            Encode_lang.append(L2_lang)
            Decode_lang.append(L1_lang)
        else:
            pairs.append([L1_lang, L2_lang])
            Encode_lang.append(L1_lang)
            Decode_lang.append(L2_lang)
    if verbose:
        print(pairs)

    return Encode_lang, Decode_lang, pairs


# Data preparation
# 상수 및 장치 설정
SOS_token = 0
EOS_token = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)

# 무작위 샘플 출력
for idx in range(10):
    print(random.choice(pairs))

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")


class Encoder(nn.Module):
    def __init__(self, vocab_size : int, embed_size : int, hidden_size, dropout = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.GRU = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.GRU(embedded)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(2)

    def forward(self, query, key):  # ← 이렇게 클래스 안에 있어야 해
        key = key
        query = query
        value = key
        Attscore = torch.bmm(query, key.permute(0, 2, 1))
        Attscore = self.softmax(Attscore)
        context = torch.bmm(Attscore, value)
        
        return context, Attscore
