import json
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from transformers import BartModel
from transformers import BartModel, BartTokenizer
from transformers import BartConfig, BartModel
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel

from tokenizers import Tokenizer
import torch
import os

num_workers = os.cpu_count() // 2  # CPU 개수의 절반 사용 (보통 최적)
print(f"추천 num_workers 값: {num_workers}")

# 현재 토크나이저 병열화와, num_workers를 사용하는 병렬화가 충동일 일으킴으로, 토크나이저 병렬화 기능을 제거하고시돟.
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# 데이터셋 정의
class DecoderDataset(Dataset):
    def __init__(self, df_path):
        df = pd.read_csv(df_path)
        self.inputs = df["input"].tolist()
        self.outputs = df["output"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        return input_text, output_text

# 디코더 네트워크 정의
class CrossAttentionDecoder(nn.Module):
    def __init__(self, 
                 kobart_tokenizer_vocab_size, 
                 hidden_dim=768, 
                 num_layers=4, 
                 num_heads=8, 
                 dropout=0.1):
        
        super(CrossAttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.kobart_tokenizer_vocab_size = kobart_tokenizer_vocab_size

        # Transformer Decoder Layer에 dropout 추가
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout 추가
        self.fc_out = nn.Linear(hidden_dim * 4, kobart_tokenizer_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, input2, tgt_mask=None):
        """
        input1: 난독화된 텍스트 임베딩 벡터 (Key, Value) -> (batch_size, seq_len, hidden_dim)
        input2: 복원된 한국어 텍스트 임베딩 벡터 (Query) -> (batch_size, seq_len, hidden_dim)
        tgt_mask: (batch_size, seq_len, seq_len) 크기의 마스크 텐서 (외부에서 제공)
        """
        decoder_output = self.decoder(input2, input1, tgt_mask=tgt_mask)

        x = self.fc1(decoder_output)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout 적용
        x = self.fc_out(x)

        return self.softmax(x)


# kobart, kobart_tokenizer 가져오는 코드
def get_kobart_and_tokenizer():
    kobart_tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
    kobart_model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
    return kobart_tokenizer, kobart_model


# mybart, mybart_tokenizer 가져오는 코드
def get_mybart_and_tokenizer(tokenizer_path, model_path, model_config):
    mybart_tokenizer = Tokenizer.from_file(tokenizer_path)
    # BART 모델 설정
    mybart = BartModel(model_config)  # BART 모델 생성

    # 저장된 textEncoder 모델 가중치 불러오기
    state_dict = torch.load(model_path, map_location="cpu")
    # BART 모델의 인코더 부분에만 가중치 로드
    mybart.encoder.load_state_dict(state_dict, strict=False)

    print("✅ BART 인코더 가중치 로드 완료!")
    return mybart_tokenizer, mybart



# input1(난독화 텍스트) 임베딩 벡터 생성용 함수 (배치 단위 지원)
def get_encodedKr_emb_vec(input_texts, mybart_tokenizer, mybart_model, device, max_length=1026):
    '''
    mybart_model는 eval() 모드여야 한다.
    input_texts: List of input strings (배치 단위)
    '''
    input_ids_batch = []
    attention_mask_batch = []

    for input_text in input_texts:
        input_encoded = mybart_tokenizer.encode(input_text)
        input_ids = input_encoded.ids[:max_length]
        attention_mask = [1] * len(input_ids)

        pad_id = mybart_tokenizer.token_to_id("<pad>")
        input_ids += [pad_id] * (max_length - len(input_ids))
        attention_mask += [0] * (max_length - len(attention_mask))

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)

    # (batch_size, seq_len) 형태로 변환
    input_ids = torch.tensor(input_ids_batch, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask_batch, dtype=torch.long).to(device)

    input_emb = mybart_model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return input_emb  # (batch_size, seq_len, hidden_dim)

# input2(한국어 텍스트) 임베딩 벡터 생성용 함수 (배치 단위 지원)
def get_korean_emb_vec(output_texts, kobart_tokenizer, kobart_model, device, max_length=1026):
    '''
    kobart_model는 eval() 모드여야 한다.
    output_texts: List of output strings (배치 단위)
    '''
    output_ids_batch = kobart_tokenizer(output_texts, return_tensors="pt", padding="max_length",
                                        max_length=max_length, truncation=True)["input_ids"].to(device)

    output_emb = kobart_model(output_ids_batch).encoder_last_hidden_state
    return output_emb  # (batch_size, seq_len, hidden_dim)


def reconstruct_text(token_ids, kobart_tokenizer):
    '''
    시퀀셜한 토큰번호 -> 원복텍스트 복구작업.
     predicted_token_ids    (batch_size, max_length) 텐서
     kobart_tokenizer       (한국어 토크나이저)
    '''
    decoded_texts = []
    batch_size = token_ids.shape[1]  # 배치 크기

    for i in range(batch_size):  # 배치 크기만큼 반복
        token_ids = token_ids[:, i].tolist()  # 🔥 (max_len,) 형태로 변환
        decoded_text = kobart_tokenizer.decode(token_ids)  # 🔥 개별 문장 복원
        decoded_texts.append(decoded_text)    
    for i, text in enumerate(decoded_texts):
        print(f"🔹 복원된 문장 {i+1}: {text}")


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def decoder_train(
        datset_path='datas/decoder_augmentation.csv',
        tokenizer_path="tokenizers/BPE_tokenizer_50000_aug.json",
        model_path="trained_encoder3.pth",
        model_save_path="trained_decoder.pth",
        batch_size=2,
        epochs=3,
        batch_max_len=1026,
        learning_rate=5e-5,
        patience=2,  # 조기 종료 기준 (연속 `patience` 횟수 동안 개선 없으면 종료)
        min_delta=0.001,  # 개선 최소 기준 (이하로 개선되면 종료)
        
        model_config=BartConfig(
            vocab_size=50000,
            d_model=768,
            encoder_layers=4,
            encoder_attention_heads=8,
            max_position_embeddings=1026,
        ),

        mydecoder=CrossAttentionDecoder(
            kobart_tokenizer_vocab_size=30000,
            hidden_dim=768,
            num_layers=1,
            num_heads=1,
            dropout=0.1
        ),
):
    # 데이터셋, 데이터로더 정의
    decoder_dataset = DecoderDataset(df_path=datset_path)
    num_workers = os.cpu_count() // 2  # ✅ 시스템에 맞는 최적의 CPU 코어 사용
    dataloader = DataLoader(
        decoder_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 모델 및 토크나이저 로드
    kobart_tokenizer, kobart_model = get_kobart_and_tokenizer()
    mybart_tokenizer, mybart_model = get_mybart_and_tokenizer(tokenizer_path, model_path, model_config)

    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kobart_model.eval(), mybart_model.eval()
    kobart_model.to(device), mybart_model.to(device)
    mydecoder.to(device)

    # 손실 함수 및 옵티마이저
    loss_fn = nn.CrossEntropyLoss(ignore_index=kobart_tokenizer.pad_token_id)
    optimizer = optim.AdamW(mydecoder.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # 학습 기록 저장용 리스트
    loss_history = []
    accuracy_history = []

    # Early Stopping 설정
    best_loss = float('inf')
    epochs_no_improve = 0

    # 학습 루프 시작
    for epoch in range(epochs):
        mydecoder.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for step, (inputs, outputs) in tqdm_bar:
            optimizer.zero_grad()

            # 🔹 배치 내 최대 `seq_len` 찾기
            input_emb = get_encodedKr_emb_vec(inputs, mybart_tokenizer, mybart_model, device).permute(1, 0, 2)
            output_emb = get_korean_emb_vec(outputs, kobart_tokenizer, kobart_model, device).permute(1, 0, 2)

            # 🔹 `tgt_mask` 차원 확장
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(batch_max_len).to(device)

            # 🔹 Mixed Precision Training 적용
            with autocast(device_type=device.type):
                decoder_output = mydecoder(input_emb, output_emb, tgt_mask=tgt_mask)

                # 🔹 정답 데이터 로드
                ground_truth_token_ids = kobart_tokenizer(outputs, return_tensors="pt", padding="max_length",
                                                          max_length=batch_max_len, truncation=True)["input_ids"].T.to(device)

                # 🔹 손실 계산
                loss = loss_fn(decoder_output.reshape(-1, decoder_output.size(-1)), ground_truth_token_ids.reshape(-1))

            # 🔹 Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 🔹 손실 및 정답률 계산
            total_loss += loss.item()

            # 🔹 예측값 & 정답 비교 (패딩 제외)
            predicted_token_ids = decoder_output.argmax(dim=-1)  # (seq_len, batch_size)
            correct = ((predicted_token_ids == ground_truth_token_ids) & (ground_truth_token_ids != kobart_tokenizer.pad_token_id)).sum().item()
            valid_tokens = (ground_truth_token_ids != kobart_tokenizer.pad_token_id).sum().item()

            total_correct += correct
            total_tokens += valid_tokens

            accuracy = correct / valid_tokens if valid_tokens > 0 else 0

            tqdm_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

        # 🔹 Early Stopping 체크
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break

    print("\n🎉 훈련 완료! 모델과 학습 기록이 저장되었습니다.")
