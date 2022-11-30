from random import random

import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.cuda import device
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


def flat_accuracy(preds, label):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = label.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))

df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")

combined = pd.concat([df1, df2, df3])
combined['class'] = combined['class'].replace(['N', 'R', 'D'], [0, 1, 2])
combined['class'] = pd.to_numeric(combined['class'], errors='coerce')
combined = combined.dropna()

combined['tweets'] = "[CLS] " + combined['tweets'] + " [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
train_token = [tokenizer.tokenize(s) for s in combined['tweets']]

input_ids = [tokenizer.convert_tokens_to_ids(s) for s in train_token]

input_ids = pad_sequences(input_ids, maxlen=256, dtype="long", truncating="post", padding="post")

masks = []

for seq in input_ids:
    seq_mask = [float(i != 0) for i in seq]
    masks.append(seq_mask)

labels = combined['class'].values
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42,
                                                                                    test_size=0.3)
train_masks, validation_masks, _, _ = train_test_split(masks, input_ids, random_state=42, test_size=0.3)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

total_steps = len(train_dataloader)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
model.zero_grad()

print("training ...\n")

total_loss = 0
model.train()

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(train_dataloader):
    # 경과 정보 표시
    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch

    # Forward 수행
    outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels)

    # 로스 구함
    loss = outputs[0]

    # 총 로스 계산
    total_loss += loss.item()

    # Backward 수행으로 그래디언트 계산
    loss.backward()

    # 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 그래디언트를 통해 가중치 파라미터 업데이트
    optimizer.step()

    # 스케줄러로 학습률 감소
    scheduler.step()

    # 그래디언트 초기화
    model.zero_grad()

# 평균 로스 계산
avg_train_loss = total_loss / len(train_dataloader)

print("")
print("  Average training loss: {0:.2f}".format(avg_train_loss))

# ========================================
#               Validation
# ========================================

print("")
print("Running Validation...")


# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for batch in validation_dataloader:
    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print("")
print("Training complete!")