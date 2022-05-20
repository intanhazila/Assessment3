# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:12:24 2022

@author: intanhazila
"""

import re
import os
import json
import datetime
import numpy as np
import pandas as pd
#from sentiment_analysis_modules import ExploratoryDataAnalysis, ModelCreation 
#from sentiment_analysis_modules import ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Embedding
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'Log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')
 
#%% EDA
# Step 1) Import data
df = pd.read_csv(URL)

df.info()
# There is 2 columns named category and text
# Both with 2225 entries data and in object datatype
df.describe()
# duplicated data is spotted. Data cleaning is needed

# Step 2) Data Cleaning
df.duplicated().sum()
# There is 99 duplicated data, we should remove it and keep the first
df_clean = df.drop_duplicates(keep='first')

text = df_clean['text']
category = df_clean['category']

# remove html tags
text_clean = [re.sub('<.*?>', '', text) for text in text]
# Convert to lower case & split
text_split = [re.sub('[^a-zA-Z]', ' ', text).lower().split() for text in text_clean]
# from text_split list, min element in the article is 123 and max is 4469

# Step 3) Features Selection
# Step 4) Data vectorization

# Tokenizer steps on the text_split

num_words=100000
oov_token='<OOV>'
prt=False

# tokenizer to vectorize the words
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(text_split)

# To save the tokenizer for deployment purpose
token_json = tokenizer.to_json()

with open(TOKEN_SAVE_PATH,'w') as json_file:
    json.dump(token_json, json_file)

# to observe the number of words
word_index = tokenizer.word_index

if prt == True:
    # to view the tokenized words
    # print(word_index)
    print(dict(list(word_index.items())[0:10]))

# to vectorize the sequences of text
token_text = tokenizer.texts_to_sequences(text_split)
        
# Pad sequence on the text        
        
tokenpad_text = pad_sequences(token_text, maxlen=500, padding='post', 
                              truncating='post')       

# Step 5) Preprocessing
#One hot encoder
one_hot_encoder = OneHotEncoder(sparse=False)
category = one_hot_encoder.fit_transform(np.expand_dims(category,axis=-1))

# Train Test Split
# x = tokenpad_text, y = category
x_train, x_test, y_train, y_test = train_test_split(data, category,
                                                    test_size = 0.3,
                                                    random_state = 123)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# before fitting into model, need to convert x_train, x_test into float
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')



print(y_train[25])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[25], axis=0)))

# [1. 0. 0. 0. 0.] - business
# [0. 1. 0. 0. 0.] - entertainment
# [0. 0. 1. 0. 0.] - politics
# [0. 0. 0. 1. 0.] - sport
# [0. 0. 0. 0. 1.] - tech

#%% model creation

model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()


log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile & model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(x_train,y_train,epochs=5,
          validation_data=(x_test,y_test), 
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Pre allocation of memory approach
predicted = np.empty([len(x_test), 2])
for index, test in enumerate(x_test):
    predicted[index,:] = model.predict(np.expand_dims(test,axis=0))

#%% Model analysis
y_pred = np.argmax(predicted, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)












