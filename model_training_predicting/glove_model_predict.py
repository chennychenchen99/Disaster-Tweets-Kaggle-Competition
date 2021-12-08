import pandas as pd
import re
import numpy as np
import tensorflow.keras.backend as K
import pickle
import matplotlib as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# -------------------------------------------------------------
file_test = '../datasets/test.csv'
df_test = pd.read_csv(file_test)

id_list = df_test['id'].to_list()

del(df_test['id'])
del(df_test['keyword'])
del(df_test['location'])

df_test['text'] = df_test['text'].str.lower()

texts_test = df_test['text'].to_list()

cleaned_texts_test = []

for text in texts_test:
    cleaned = re.sub(r'https?://\S+', '', text)
    cleaned = re.sub(r'\n',' ', cleaned)
    cleaned = re.sub('\s+', ' ', cleaned).strip()
    cleaned = re.sub('[\W]+', ' ', cleaned)
    # remove emojis
    emojis = re.compile("["
                        u"\U0001F600-\U0001F64F"
                        u"\U0001F300-\U0001F5FF"
                        u"\U0001F680-\U0001F6FF"
                        u"\U0001F1E0-\U0001F1FF"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251]+", flags=re.UNICODE)
    cleaned = emojis.sub(r'', cleaned)

    cleaned_texts_test.append(cleaned)

# --------------------------------------------------------------------

# Loading the Tokenizer
max_len = 25

tokenizer = None
with open('../models/glove_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(cleaned_texts_test)

word_index = tokenizer.word_index

test_data = pad_sequences(sequences, maxlen=max_len)

# -------------------- Using the Model --------------------------------

model = load_model('../models/glove_model.h5', custom_objects={'get_f1': get_f1})

preds = model.predict(test_data)
output_preds = []

for pred in preds:
    if pred >= 0.50:
        output_preds.append(1)
    else:
        output_preds.append(0)


f = open('../submissions/DanielChen_glove_submission.csv', 'w')
f.write("id,target\n")
for i in range(len(id_list)):
    f.write("{},{}\n".format(id_list[i], output_preds[i]))
f.close()
