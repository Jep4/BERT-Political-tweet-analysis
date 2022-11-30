import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)


df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")

train = pd.concat([df1, df2])
test = df3


def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tf.reduce_mean(embeddings, 1))


list_train = [train[i:i + 100] for i in range(0, train.shape[0], 100)]
list_test = [test[i:i + 100] for i in range(0, test.shape[0], 100)]
elmo_train = [elmo_vectors(x['tweets']) for x in train]
elmo_test = [elmo_vectors(x['tweets']) for x in test]

elmo_train_new = np.concatenate(elmo_train, axis=0)
elmo_test_new = np.concatenate(elmo_test, axis=0)

xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new,
                                                  train['label'],
                                                  random_state=42,
                                                  test_size=0.2)

lreg = LogisticRegression()
lreg.fit(xtrain, ytrain)
preds_valid = lreg.predict(xvalid)
f1_score(yvalid, preds_valid)
