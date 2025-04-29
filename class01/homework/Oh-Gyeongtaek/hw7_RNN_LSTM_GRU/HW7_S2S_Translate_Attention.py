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
device = "cuda" if torch.cuda.is_available() else "cpu"

lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")

class Encoder(nn.Module):
    def __init__(self,
                vocab_size: int,
                embed_size: int,
                hidden_size,
                dropout = 0.1,
                ):
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
        
    def forward(self, query, key):
        key = key
        query = query
        value = key
        Attscore = torch.bmm(query, key.permute(0, 2, 1))
        Attscore = self.softmax(Attscore)
        context = torch.bmm(Attscore, value)
        
        return context, Attscore

class DecoderAttention(nn.Module):

    def forward_1_step(self, input_word, hidden, hidden_from_encoder):
        embedded_input_word = self.embedding(input_word)
        
        output, hidden1 = self.GRU(embedded_input_word, hidden)
        context, att_score = self.attention(output, hidden_from_encoder)
        concat = torch.cat((output, context), dim=2)
        combined = torch.tanh(self.attn_combine(concat.squeeze(1)))
        
        logits = self.out(combined)
        sel_tok = F.softmax(logits, dim=-1)
        return sel_tok.unsqueeze(1), hidden_new, attn_weights

    def __init__(self,
                vocab_size : int,
                embed_size : int,
                hidden_size : int,
                max_length = 100,
                device = 'cuda'
                ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, 
                               padding_idx=tokenizer.pad_token_id).to(device)
        self.attention = Attention(hidden_size).to(device)
        self.GRU = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size).to(device)
        self.attn_combine = nn.Linear(2*hidden_size, hidden_size).to(device)
        
        self.max_len = max_length
        self.device = device
        self.to(device)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, 
            teacher_forcing_ratio=0.5):
    
        batch_size = encoder_outputs.size(0)
        if target_tensor is not None:
            tgt_len = target_tensor.size(1)
        else:
            tgt_len = self.max_len
        vocab_size = self.out.out_features

    