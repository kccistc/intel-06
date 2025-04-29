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
print(device) # cuda인지 cpu로 작동되는지 확인
lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")

class Encoder(nn.Module):
    def _init_(self,
                # moduleSelect = int,
                vocab_size : int,
                embed_size : int,
                hidden_size ,
                dropout = 0.1
                ):
        super()._init_()

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

class DecoderAttention(nn.Moudule):
    def forward_1_step(self, input_word, hidden, hidden_from_encoder):
        # Query (batch, sentance_len, word_vec) ====> (batch, word_vec, setance_len)
        # output (batch, sentance_len, word_vec)
        # attention(query, key)
        embedded_input_word = self.embedding(input_word)
        output, hidden1 = self.GRU(embedded_input_word, hidden)
        context, att_score = self.attention(output, hidden_from_encoder)
        concat = torch.cat((output, context), dim=2)
        combined = torch.tanh(self.attn_combine(concat.squeeze(1)))

        logits = self.out(combined)
        sel_tok = F.softmax(logits, dim=-1)
        return sel_tok.unsqueeze(1), hidden_new, attn_weights

class DecoderAttention(nn.Module):
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

class DecoderAttention(nn.Module):
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5):
        """
        encoder_outputs: (batch, src_len, Hidden)
        encoder_hidden: (1, batch, Hidden)
        target_tensor:
        (batch, tgt_len)
        """
        batch_size = encoder_outputs.size(0)
        if target_tensor is not None:
            tgt_len = target_tensor.size(1) # sentance_length
        else:
            tgt_len = self.max_len
        # 출력 벡터의 차원 수 (in_feature는 입력 벡터의 차원 수)
        vocab_size = self.out.out_features

class DecoderAttention(nn.Module):
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5):
        # 이전 페이지 에서 이어짐.
        # sentance 길이만큼 반복해야 함.
            for sent_idx in range(tgt_len):
                sel_tok = self.forward_1_step(decoder_input, decoder_hidden, encoder_outputs)
                # 저장.
            outputs[:, sent_idx, :] = sel_tok.squeeze(1)
            # 다음 입력 처리
            # teacher forcing을 통해 정답을 넣을지 예측을 넣을지 정함.
            if (target_tensor is not None and
                random.random() < teacher_forcing_ratio):
                # 정답 입력
                decoder_input = target_tensor[:, sent_idx].unsqueeze(1)
            else:
                #예측 입력
                top1 = sel_tok.argmax(dim=2)
                decoder_input = top1
    return outputs

# Layer의 크기 선언.
embed_size = 16
hidden_size = 16
src_vocab = tokenizer.vocab_size
tgt_vocab = tokenizer.vocab_size
encoder = Encoder(src_vocab, embed_size, hidden_size).to(device)
decoder = DecoderAttention(tgt_vocab, embed_size, hidden_size,
max_length=target_ids.size(1), device=device).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) +
list(decoder.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# 학습실행
num_epochs = 100
for epoch in range(1, num_epochs+1):
    encoder.train(); decoder.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for src_ids, tgt_ids in loop:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        optimizer.zero_grad()
        enc_outs, enc_hidden = encoder(src_ids)
        # dec_output (batch, tgt_len, vocab)
        dec_outputs = decoder(enc_outs, enc_hidden, target_tensor=tgt_ids)
        batch_loss = 0
        for t in range(tgt_ids.size(1)):
            batch_loss += criterion(dec_outputs[:, t, :], tgt_ids[:, t])
        batch_loss = batch_loss / tgt_ids.size(1)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        loop.set_postfix(loss=batch_loss.item())
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:2d} Loss: {total_loss:.4f}")