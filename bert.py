
import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModel
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

combined = pd.concat([df1, df2])
combined['class'] = combined['class'].replace(['N', 'R', 'D'], [0, 1, 2])
combined['class'] = pd.to_numeric(combined['class'], errors='coerce')
combined = combined.dropna()

# TOKENIZE ================

# input_ids = tokenized data

combined['tweets'] = "[CLS] " + combined['tweets'] + " [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
train_token = [tokenizer.tokenize(s) for s in combined['tweets']]

input_ids = [tokenizer.convert_tokens_to_ids(s) for s in train_token]

input_ids = pad_sequences(input_ids, maxlen=256, dtype="long", truncating="post", padding="post")

# masks = array of 1 & 0
# 0 for value 0 in input_ids
# to neglect unimportant data

masks = []

for seq in input_ids:
    seq_mask = [float(i != 0) for i in seq]
    masks.append(seq_mask)

# Split train & test values ==============
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

# import BERT model ==============
model = AutoModel.from_pretrained('bert-base-uncased')

# List to tensor ====================
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# TEST SET ============================
test = df3
df3['class'] = df3['class'].replace(['N', 'R', 'D'], [0, 1, 2])
df3['class'] = pd.to_numeric(df3['class'], errors='coerce')
df3 = df3.dropna()


df3['tweets'] = "[CLS] " + df3['tweets'] + " [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
train_token = [tokenizer.tokenize(s) for s in df3['tweets']]

input_ids = [tokenizer.convert_tokens_to_ids(s) for s in train_token]
input_ids = pad_sequences(input_ids, maxlen=256, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

labels = df3['class'].values

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)
batch_size = 32
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# ===============================
