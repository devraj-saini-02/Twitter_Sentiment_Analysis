import numpy as np 
import pandas as pd 
df=pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_training.csv',names=cols)
df = df.dropna()
xt_initial=df['tweet']
yt=df['label']
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xt_initial)
xt_initial = tokenizer.texts_to_sequences(xt_initial)
xt_initial[0]
from tensorflow.keras.utils import pad_sequences
xt_padded = pad_sequences(xt_initial,padding='post')
xt_padded[0]
maxl=len(xt_padded[0])
maxl
total_words=len(tokenizer.word_index)+1
total_words
mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2,'Irrelevant':3}
yt = df['label'].map(mapping).values
df_valid=pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_validation.csv',names=cols)
df_valid = df_valid.dropna()
xv_initial=df_valid['tweet']
yv=df_valid['label']
xv_initial = tokenizer.texts_to_sequences(xv_initial)
xv_padded = pad_sequences(xv_initial, maxlen=maxl, padding='post')
yv = yv.map(mapping).values
import keras
from keras import Sequential
from keras.layers import Dense,Embedding,Flatten,GRU,Dropout,LSTM,SimpleRNN

#GRU MODEL - Better results + computationally efficient
#SimpleRNN MODEL- Computationally efficient
#LSTM Model - Best Performance

model=Sequential()
model.add(Embedding(total_words, output_dim=60, mask_zero=True))##mask_zero helps in improving accuracy
model.add(SimpleRNN(64, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(4, activation='softmax'))

from keras.optimizers import Adam

model.compile( optimizer=Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy'
,metrics=['accuracy'] )

from keras.callbacks import ModelCheckpoint as MCP
checkpoint=MCP('best_model.h5',
               monitor='val_accuracy',
              save_best_only=True,
              mode='max',
              verbose= 1)

history = model.fit(
    xt_padded, yt,          
    validation_data=(xv_padded, yv), 
    epochs=8,
    batch_size=32,
    callbacks=[checkpoint]
)

from keras.models import load_model
best_model = load_model('best_model.h5')

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

loss, acc = best_model.evaluate(xv_padded, yv)

y_pred_probs = best_model.predict(xv_padded)
y_pred = np.argmax(y_pred_probs, axis=1)

if yv.ndim > 1:  
    y_true = np.argmax(yv, axis=1)
else:
    y_true = yv

cm = confusion_matrix(y_true, y_pred)
labels = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=labels))
