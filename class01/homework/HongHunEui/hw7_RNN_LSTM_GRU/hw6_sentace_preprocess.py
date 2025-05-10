import re
import unicodedata
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

# 유니코드 -> ASCII (한글은 유지, 영어는 결합 문자 제거)
def unicodeToAscii(s):
    s = re.sub(r'(?i)cc-by.*$', '', s)
    hangul_pattern = re.compile('[가-힣ㄱ-ㅎㅏ-ㅣ]')
    result = []
    for c in s:
        if hangul_pattern.match(c):
            result.append(c)
        else:
            for c_ in unicodedata.normalize('NFD', c):  # 결합문자 분해
                if unicodedata.category(c_) != 'Mn':     # 결합문자 제거
                    result.append(c_)
    return ''.join(result)

# 문자열 정규화
def Norm_String(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ!?]+", r" ", s)
    return s.strip()

# 병렬 말뭉치 로드
def read_language(L1, L2, reverse=False, verbose=False):
    print("read languages data...")
    pairs = []
    Encode_lang = []
    Decode_lang = []
    pf = open('%s2%s.txt' % (L1, L2), encoding='utf-8').read().strip().split('\n')
    for ll in pf:
        parts = ll.split('\t')
        if len(parts) > 1:
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
