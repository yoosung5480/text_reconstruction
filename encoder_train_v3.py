import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig, BartModel
from torch.optim import AdamW
from tqdm import tqdm
import random
'''
v2까지는 인코더 훈련과정에 쓸대없는 ƒc층이 포함돼서 인코더의 범주를 넘기는 훈련을 시켰다.
새로이 정의할 v3코드에서는 임베딩 벡터 생성훈련에만 주목한다. 
'''

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
    tokens = text.split()
    num_masks = max(1, int(len(tokens) * mask_prob))
    mask_positions = random.sample(range(len(tokens)), num_masks)
    for pos in mask_positions:
        mask_length = random.randint(1, max_mask_size)
        tokens[pos:pos + mask_length] = [mask_token]
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
        input_encoded = self.tokenizer.encode(input_text)
        target_encoded = self.tokenizer.encode(target_text)

        input_ids = input_encoded.ids[:self.max_len]
        target_ids = target_encoded.ids[:self.max_len]
        attention_mask = [1] * len(input_ids)

        pad_id = self.tokenizer.token_to_id("<pad>")
        input_ids += [pad_id] * (self.max_len - len(input_ids))
        target_ids += [pad_id] * (self.max_len - len(target_ids))
        attention_mask += [0] * (self.max_len - len(attention_mask))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }

# 드롭아웃을 적용해보자!
class textEncoder(nn.Module):
    def __init__(self, config):
        super(textEncoder, self).__init__()
        self.bart = BartModel(config)  # 그대로 유지 (FC 레이어 삭제)

    def forward(self, input_ids, attention_mask):
        outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # FC 레이어 없이 임베딩 벡터 출력


def train(
        train_path='datas/train.csv',
        save_path="trained_encoder.pth",
        tokenizer_path="tokenizers/BPE_tokenizer_50000_aug.json",
        epochs=20,  # 초기 값 증가 (조기 종료 적용)
        batch_size=8,
        sample_size=500,
        d_model=768,
        encoder_layers=3,
        encoder_attention_heads=4,
        patience=3,  # Early Stopping 기준 (N 에포크 동안 개선 없으면 종료)
        min_delta=0.0001  # 손실 감소 기준
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if sample_size == -1:
        train_pd = pd.read_csv(train_path)
    else:
        train_pd = pd.read_csv(train_path).sample(sample_size)

    train_text = list(train_pd['input'])
    masked_text = [apply_infilling_masking(x) for x in train_text]

    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = EncoderDataset(masked_text, train_text, tokenizer=tokenizer, max_len=1024)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        max_position_embeddings=1024
    )

    encoder = textEncoder(config=config).to(device)
    scaler = amp.GradScaler()
    optimizer = AdamW(encoder.parameters(), lr=5e-5)
    loss_func = nn.MSELoss()

    losses = []
    best_loss = float('inf')  # 최소 손실 초기화
    patience_counter = 0  # 조기 종료 카운터

    for epoch in range(epochs):
        encoder.train()
        total_loss = .0

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type=device.type):
                predicted_embeddings = encoder(input_ids, attention_mask)
                target_embeddings = encoder(labels, attention_mask)
                loss = loss_func(predicted_embeddings, target_embeddings)

            if torch.isnan(loss):
                print(f"NaN loss detected at step {idx}, skipping...")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # 조기 종료 로직
        if avg_loss < best_loss - min_delta:  # 손실이 개선되었는지 확인
            best_loss = avg_loss
            patience_counter = 0  # 개선되었으므로 카운터 초기화
            torch.save(encoder.state_dict(), save_path)  # 개선될 때마다 모델 저장
            print(f"🔹 Model improved. Saved at {save_path}.")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter}/{patience} epochs.")

        if patience_counter >= patience:
            print("⏹ Early Stopping triggered. Training stopped.")
            break  # 조기 종료

    # 손실 그래프 출력
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print(f"훈련된 모델이 '{save_path}'로 저장되었습니다.")

    return encoder
