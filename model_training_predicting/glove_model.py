import os
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import re
import tensorflow_hub as hub
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# --------------------- Data Preprocessing ----------------------------

file = '../datasets/train.csv'
df = pd.read_csv(file)

del(df['id'])
del(df['keyword'])
del(df['location'])

df_attributes = df.iloc[:, :-1]
df_label = df.iloc[:, -1:]

df_attributes['text'] = df_attributes['text'].str.lower()

texts = df_attributes['text'].to_list()
training_labels = df_label['target'].to_list()

cleaned_texts = []

for text in texts:
    cleaned = re.sub(r'https?://\S+', '', text)
    cleaned = re.sub(r'\n',' ', cleaned)
    cleaned = re.sub('\s+', ' ', cleaned).strip()
    cleaned = re.sub('[\W]+', ' ', cleaned)

    emojis = re.compile("["
                        u"\U0001F600-\U0001F64F"
                        u"\U0001F300-\U0001F5FF"
                        u"\U0001F680-\U0001F6FF"
                        u"\U0001F1E0-\U0001F1FF"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251]+", flags=re.UNICODE)
    cleaned = emojis.sub(r'', cleaned)

    cleaned_texts.append(cleaned)


# ----------------------------------------------------------------

# # Find the distribution of word length in each Tweet
seq_lengths = np.asarray([len(s.split()) for s in cleaned_texts])
print([(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])

# vocab = []
# for text in cleaned_texts:
#     split = text.split()
#     for word in split:
#         if word not in vocab:
#             vocab.append(word)
#
# # Num of vocab words is 17076
# print("Vocab size: {}".format(len(vocab)))
# ----------------------------------------------------------------

# Tokenizing the text
max_len = 25
max_words = 30000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(cleaned_texts)
sequences = tokenizer.texts_to_sequences(cleaned_texts)

word_index = tokenizer.word_index

training_data = pad_sequences(sequences, maxlen=max_len, padding='post')
training_labels = np.asarray(training_labels)

# ----------------------------------------------------------------

# Preparing the GloVe word-embedding
glove_dir = '../glove/glove.twitter.27B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.twitter.27B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# --------------------- Building the Model --------------------------

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len, weights=[
    embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizers='adamax', loss='binary_crossentropy', metrics=[
    'accuracy', get_f1])


# Saves the model with the highest f1-score
callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath='../models/glove_model.h5',
        monitor='get_f1',
        mode='max',
        save_best_only=True,
    )
]


history = model.fit(training_data, training_labels, epochs=5, batch_size=32,
                    validation_split=0.15, callbacks=callbacks_list)

# Save the tokenizer into a pickle file
with open('../models/glove_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


