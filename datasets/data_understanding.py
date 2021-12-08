import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------- Data Preprocessing ----------------------------
file = './train.csv'
df = pd.read_csv(file)

del(df['id'])

df_attributes = df.iloc[:, :-1]
df_label = df.iloc[:, -1:]

# print(df_attributes.describe())
print(df_label.describe())

count = 0
for row in df_label['target']:
    if row == 1:
        count += 1

print(count)

# print(df.columns)

# Total data instances in training: 7613, 7552 keywords, 5080 location
# 110 duplicate texts, most common duplicated text only occurs 10 times
# 57% 0's, 43% 1's

# df_duplicates = pd.DataFrame()
# df_duplicates = df_duplicates[df.duplicated(subset='text')]
# print("Duplicates in the DataFrame: \n{}".format(df_duplicates[:2]))


# Test data has 3263 instances
# f = './test.csv'
# df_test = pd.read_csv(f)
#
# del(df_test['id'])
# del(df_test['keyword'])
# del(df_test['location'])
#
# print(df_test.describe())
