from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
import numpy as np
# Load the IMDB dataset
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size) 

# Decode reviews to text for preprocessing 

word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}
decoded_reviews = [' '.join([index_word.get(i - 3, '?') for i in review]) for review in X_train[:5]]

# Pad sequences

X_train = pad_sequences(X_train, maxlen=200, padding='post')
X_test = pad_sequences(X_test, maxlen=200, padding='post')
#Load GloVe embeddings
embedding_index = {}
glove_path = 'glove.6B.100d.txt'  # Path to GloVe file

with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

#Prepare embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
# Build LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2,verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

LSTM_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
LSTM_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
LSTM_model.summary()
LSTM_history = LSTM_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
LSTM_loss, LSTM_accuracy = LSTM_model.evaluate(X_test, y_test)

print(f'Test Loss: {LSTM_loss:.4f}, Test Accuracy: {LSTM_accuracy:.4f}')

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
models = ['LSTM with GloVe', 'LSTM without GloVe']
accuracies = [accuracy, LSTM_accuracy]
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
