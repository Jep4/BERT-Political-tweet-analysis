import random
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


n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))


# READ DATA ================ 

df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")

combined = pd.concat([df1, df2, df3])
combined['class'] = combined['class'].replace(['N', 'R', 'D'], [0, 1, 2])
combined['class'] = pd.to_numeric(combined['class'], errors='coerce')
combined = combined.dropna()


# TOKENIZE ================

# input_ids = tokenized data

# masks = array of 1 & 0 
# 0 for value 0 in input_ids 
# to neglect unimportant data
combined['tweets'] = "[CLS] " + combined['tweets'] + " [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
train_token = [tokenizer.tokenize(s) for s in combined['tweets']]

input_ids = [tokenizer.convert_tokens_to_ids(s) for s in train_token]

input_ids = pad_sequences(input_ids, maxlen=256, dtype="long", truncating="post", padding="post")

masks = []

for seq in input_ids:
    seq_mask = [float(i != 0) for i in seq]
    masks.append(seq_mask)


# Split train & test values ============== 
labels = combined['class'].values
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42,
                                                                                    test_size=0.3)
train_masks, validation_masks, _, _ = train_test_split(masks, input_ids, random_state=42, test_size=0.3)

# import BERT model ==============
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()
