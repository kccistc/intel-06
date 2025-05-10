import re
import unicodedata
import torch
import torch.nn as nn
import numpy as np
import random
from sentance_preprocess import read_language
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

SOS_token = 0
EOS_token = 1
# MAX_LENGTH = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")


class Encoder(nn.Module):
    def __init__(self,

        # moduleSelect = int,
        vocab_size : int,
        embed_size : int,
        hidden_size ,
        dropout = 0.1,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=tokenizer.pad_token_id)
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
        # decoder input으로는 문자가 들어오니, hidden size (wordvec 을 알고있어야 함)
def forward(self, query, key):
    # input: decoder
    # hidden: encoder
    key = key
    query = query
    value = key
    Attscore = torch.bmm(query, key.permute(0, 2, 1))
    # softmax,
    Attscore = self.softmax(Attscore)
    context = torch.bmm(Attscore, value)

    return context, Attscore