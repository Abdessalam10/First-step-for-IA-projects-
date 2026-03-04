from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

vocab_size = 10000
maxlen = 200


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=maxlen,padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen,padding='post')

model = Sequential([
    Embedding(input_dim = vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.summary()

#history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#loss, accuracy = model.evaluate(X_test, y_test)
#print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=maxlen,padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen,padding='post')
train_dataset = TensorDataset(torch.tensor(X_train ), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        output = self.fc(hidden[-1])
        return self.sigmoid(output)
    
model = RNNModel(vocab_size,embedding_dim =128,hidden_dim= 128, output_dim= 1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        
train(model, train_loader, criterion, optimizer, epochs=5) 

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test)
        y_test_tensor = torch.tensor(y_test)
        outputs = model(X_test_tensor).squeeze()
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}') 