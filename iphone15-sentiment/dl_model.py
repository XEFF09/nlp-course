import pandas as pd
from config import *
from clean import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional, SimpleRNN, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau

BATCH_SIZE = 32
EPOCHS = 10
KERNEL_SIZE = 3
FACTOR = 0.4

# Load data
df_0 = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%200')
df_1 = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%201')
data = pd.concat([df_0, df_1], ignore_index=True)

# Preprocess data
x = data['text'].astype(str).apply(big_cleaning)
y = data['class'].astype(int)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)

x_tts = tokenizer.texts_to_sequences(x)
maxlen = max([len(s) for s in x_tts])
x_pad = pad_sequences(x_tts, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.3, random_state=11)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
model.add(Conv1D(64, kernel_size=KERNEL_SIZE, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=KERNEL_SIZE, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=FACTOR, patience=20, min_lr=0.0001)
model.fit(X_train, y_train_encoded, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=[reduce_lr])

# Evaluate the model
def evaluate_model(model, X_test, y_test_encoded):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("Sentiment Analysis")
    print(confusion_matrix(y_test_encoded, y_pred_binary))
    print(classification_report(y_test_encoded, y_pred_binary, zero_division=1))
    print(accuracy_score(y_test_encoded, y_pred_binary))

# Call the evaluate function
evaluate_model(model, X_test, y_test_encoded)









# Test with Sentiment Phrases
test_phrases = pd.Series(["ผมว่าก็ไม่แย่นะครับ แต่โคตรแย่"]).astype(str).apply(big_cleaning)
tts = tokenizer.texts_to_sequences(test_phrases)
tts = pad_sequences(tts, maxlen=maxlen)

sentiment_prob = model.predict(tts)
predicted_sentiment = ["positive" if prob >= 0.5 else "negative" for prob in sentiment_prob]

print("Sentiment Prediction:", predicted_sentiment)