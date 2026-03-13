from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,GRU

vocab_size = 10000
maxlen = 200


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=maxlen,padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen,padding='post')

model = Sequential([
    Embedding(input_dim = vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

GRU_model = Sequential([
    Embedding(input_dim = vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

GRU_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

GRU_model.summary()

GRU_history = GRU_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

GRU_loss, GRU_accuracy = GRU_model.evaluate(X_test, y_test)
print(f'Test Loss: {GRU_loss:.4f}, Test Accuracy: {GRU_accuracy:.4f}')
