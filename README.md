# fake-news-detection
## Keras Model building for fake news dataset

Importing the dataset

Fake news dataset

True news dataset

Importing the required libraries used for data analytics, visualization and model building

---

import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 

import tensorflow as tf 

import re 

from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import seaborn as sns 

---

2.checking for the null values

fake_df.isnull().sum()

real_df.isnull().sum()

3.Data visualization


![image](https://user-images.githubusercontent.com/100121721/166867215-4ca73eda-984a-4a94-863d-06d85ac4d379.png)

---

plt.figure(figsize=(10, 5))

plt.bar('Fake News', len(fake_df), color='SkyBlue')

plt.bar('Real News', len(real_df), color='PeachPuff')

plt.title('Distribution of Fake News and Real News', size=15)

plt.xlabel('News Type', size=15)

plt.ylabel('# of News Articles', size=15)

---


4.Combining both the datasets

news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)

news_df

##Merging the dataset to make it ready for the test and train

6.splitting the dataset

features = news_df['text']

targets = news_df['class']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)

7.Normalizing the dataset

def normalize(data):

    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

X_train = normalize(X_train)

X_test = normalize(X_test)

8.Model building - using keras

max_vocab = 10000

tokenizer = Tokenizer(num_words=max_vocab)

tokenizer.fit_on_texts(X_train)

--- 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.summary()

---

9.Model fitting

##We are going to use early stop, which stops when the validation loss no longer improve.

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])


10.Model evaluating

model.evaluate(X_test, y_test)


11.Visualization of training and validation accuracy

history_dict = history.history

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']

loss = history_dict['loss']

val_loss = history_dict['val_loss']

epochs = history.epoch

plt.figure(figsize=(10,5))

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss', size=10)

plt.xlabel('Epochs', size=10)

plt.ylabel('Loss', size=10)

plt.legend(prop={'size': 10})

plt.show()

![image](https://user-images.githubusercontent.com/100121721/166866999-67512e5c-78c9-48b9-8b17-f7592a9cd1c1.png)

---

plt.figure(figsize=(10,5))

plt.plot(epochs, acc, 'g', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy', size=10)

plt.xlabel('Epochs', size=10)

plt.ylabel('Accuracy', size=10)

plt.legend(prop={'size': 10})

plt.ylim((0.5,1))

plt.show()

---
![image](https://user-images.githubusercontent.com/100121721/166866982-f450d4b6-3014-4e94-8602-35747b31957f.png)


12.Confision matrix

matrix = confusion_matrix(binary_predictions, y_test, normalize='all')

plt.figure(figsize=(10, 5))

ax= plt.subplot()

sns.heatmap(matrix, annot=True, ax = ax)

##labels, title and ticks

ax.set_xlabel('Predicted Labels', size=10)

ax.set_ylabel('True Labels', size=10)

ax.set_title('Confusion Matrix', size=10) 

ax.xaxis.set_ticklabels([0,1], size=10)

ax.yaxis.set_ticklabels([0,1], size=10)

![image](https://user-images.githubusercontent.com/100121721/166867100-f568903a-5af8-4e45-986b-81db9b2aaefb.png)


---
