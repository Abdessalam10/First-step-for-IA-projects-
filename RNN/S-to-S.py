import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 1. Dataset and Vocab setup
english_sentences = ["hello", "how are you", "good morning"]
french_sentences = ["bonjour", "comment ça va", "bon matin"]

def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

english_vocab = build_vocab(english_sentences)
french_vocab = build_vocab(french_sentences)

def tokenize_and_pad(sentences, vocab, max_len):
    tokenized = []
    for sentence in sentences:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        # Ensure padding doesn't exceed max_len
        tokens = tokens[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(tokens))
        tokenized.append(tokens)
    return np.array(tokenized)

max_len_eng = max(len(s.split()) for s in english_sentences) + 2
max_len_fr = max(len(s.split()) for s in french_sentences) + 2

english_data = tokenize_and_pad(english_sentences, english_vocab, max_len_eng)
french_data = tokenize_and_pad(french_sentences, french_vocab, max_len_fr)

class TranslationDataset(Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data
    def __len__(self):
        return len(self.source_data)
    def __getitem__(self, idx):
        return torch.tensor(self.source_data[idx]), torch.tensor(self.target_data[idx])

# Use lowercase 'dataset' to avoid overwriting the class 'Dataset'
train_dataset = TranslationDataset(english_data, french_data)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 2. Model Architecture
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1) 
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

# Seq2Seq is now OUTSIDE the Decoder class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
        return outputs

# 3. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = len(english_vocab)
output_dim = len(french_vocab)
emb_dim = 128
hid_dim = 512
num_layers = 2

encoder = Encoder(input_dim, emb_dim, hid_dim, num_layers).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])

def train(model, dataloader, optimizer, criterion, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}')

# 4. Translation and Execution
def translate(model, sentence, english_vocab, french_vocab, device, max_len=10):
    model.eval()
    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        
    tgt_vocab_inv = {v: k for k, v in french_vocab.items()}
    tgt_indices = [french_vocab["<SOS>"]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(device)
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
        pred = output.argmax(1).item()
        tgt_indices.append(pred)
        if pred == french_vocab["<EOS>"]:
            break
            
    translated_sentence = ' '.join([tgt_vocab_inv[idx] for idx in tgt_indices if idx not in [1, 2, 0]])
    return translated_sentence

# Run it
train(model, dataloader, optimizer, criterion, device)
test_sentence = "good morning"
translated = translate(model, test_sentence, english_vocab, french_vocab, device)
print(f'\nEnglish: {test_sentence} -> French: {translated}')
