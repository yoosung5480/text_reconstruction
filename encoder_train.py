import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbs
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.amp as amp  # AMP 모듈 추가

from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig, BartModel, PreTrainedTokenizerFast
from torch.optim import AdamW
from transformers import BartConfig, BartModel, PreTrainedTokenizerFast
from tqdm import tqdm

import random
import matplotlib.pyplot as plt




def apply_infilling_masking(text, mask_token="<mask>", mask_prob=0.15, max_mask_size=3):
    """
    Infilling Masking을 적용하는 함수.
    
    Args:
        text (str): 원본 텍스트
        mask_token (str): 마스킹 토큰 (디폴트: <mask>)
        mask_prob (float): 토큰을 마스킹할 확률 (디폴트: 0.15)
        max_mask_size (int): 최대 연속 마스킹 토큰 수 (디폴트: 3)

    Returns:
        str: 마스킹된 텍스트
    """
    # 텍스트를 공백 기준으로 토큰화
    tokens = text.split()

    # 마스킹 대상 토큰 선택
    num_masks = max(1, int(len(tokens) * mask_prob))
    mask_positions = random.sample(range(len(tokens)), num_masks)

    # 마스킹 적용
    for pos in mask_positions:
        mask_length = random.randint(1, max_mask_size)  # 연속 마스크 길이
        tokens[pos:pos + mask_length] = [mask_token]

    # 마스킹된 텍스트 반환
    return " ".join(tokens)


class EncoderDataset(Dataset):
    """
    inputs ('str' list): Text infilling된 난독화 텍스트 리스트
    targets ('str' list): 원본 난독화 텍스트 리스트
    tokenizer : 커스텀 토크나이저 (BPE, WordPiece 등)
    max_len : 원본 문자열 최대 길이
    """
    def __init__(self, inputs, targets, tokenizer, max_len):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]

        # 입력 텍스트 토큰화 (타겟 텍스트는 별도로 사용 안 함)
        input_encoded = self.tokenizer.encode(input_text)
        target_encoded = self.tokenizer.encode(target_text)

        # 토큰 ID와 패딩 적용
        input_ids = input_encoded.ids
        target_ids = target_encoded.ids
        attention_mask = [1] * len(input_ids)

        # 시퀀스 길이 조정
        if len(input_ids) < self.max_len:
            # 패딩 추가
            pad_length = self.max_len - len(input_ids)
            target_pad_length = self.max_len - len(target_ids)
            input_ids += [self.tokenizer.token_to_id("<pad>")] * pad_length
            target_ids += [self.tokenizer.token_to_id("<pad>")] * target_pad_length
            attention_mask += [0] * pad_length
        else:
            # 길이 초과 시 자르기
            input_ids = input_ids[:self.max_len]
            target_ids = target_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        # 텐서로 변환
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": target_ids.squeeze(0)
        }
    
class textEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super(textEncoder, self).__init__()
        self.bart = BartModel(config)
        # d_model = 512로 가정. 트랜스포머 논문의 ffn층과 똑같이 설정.
        # torch.Size([batch_size, seq_len, config.d_model]) -> torch.Size([batch_size*seq_len, 4*config.d_model])
        self.fc1 = nn.Linear(config.d_model, config.d_model*4)
        self.relu = nn.ReLU()
        # torch.Size([batch_size*seq_len, 4*config.d_model]) -> torch.Size([batch_size*seq_len, vocab_size])
        self.fc2 = nn.Linear(config.d_model*4, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.fc1(outputs.last_hidden_state)  # torch.Size([batch_size, seq_len, vocab_size])
        x = self.relu(x)
        x = self.fc2(x)
        logits = x.view(-1, x.size(-1))
        return logits
        



def train(
        test_path = 'datas/test.csv', 
        train_path = 'datas/train.csv',
        epochs = 4,
        batch_size = 4
        ):
    
    # 데이터셋 가져오기.
    test_pd = pd.read_csv(test_path).sample(100)
    train_pd = pd.read_csv(train_path).sample(100)

    # 난독화된 텍스트 생성
    train_text = list(train_pd['input'])
    masked_text = [apply_infilling_masking(x) for x in train_text]
    textGT = list(train_pd['output'])

    # 저장된 토크나이저 로드
    tokenizer = Tokenizer.from_file("tokenizers/BPE_tokenizer_50000_aug.json")

    # 데이터셋 정의
    dataset = EncoderDataset(masked_text, train_text, tokenizer=tokenizer, max_len=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    # BART 모델 생성
    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        encoder_layers=3,
        encoder_attention_heads=4,
        max_position_embeddings=2000
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = textEncoder(config=config, vocab_size=tokenizer.get_vocab_size()).to(device)
    
    # 16비트 AMP 설정
    scaler = amp.GradScaler()  # Gradient Scaler 추가

    # 훈련 설정
    optimizer = AdamW(encoder.parameters(), lr=5e-5)
    loss_func = nn.CrossEntropyLoss()
    
    # 손실 및 정확도 기록
    losses = []
    accuracies = []

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.view(-1)

            optimizer.zero_grad()

            # 16비트 연산 적용
            with amp.autocast(device_type=device.type):
                predicted_logits = encoder(input_ids, attention_mask)
                loss = loss_func(predicted_logits, labels)

            # Mixed Precision Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # 정확도 계산
            predicted_tokens = torch.argmax(predicted_logits, dim=-1)  # 가장 높은 확률 토큰 선택
            correct = (predicted_tokens == labels).sum().item()  # 맞춘 개수
            total_correct += correct
            total_tokens += labels.numel()  # 전체 토큰 개수

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens

        losses.append(avg_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # 손실 및 정확도 그래프 출력
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(accuracies, label="Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # 훈련된 모델 저장
    torch.save(encoder.state_dict(), "trained_encoder.pth")
    print("훈련된 모델이 'trained_encoder.pth'로 저장되었습니다.")
