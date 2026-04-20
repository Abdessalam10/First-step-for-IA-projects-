import numpy as np

#define query, key, value
def generate_data(seq_len, embed_dim):
    np.random.seed(42)
    return np.random.rand(seq_len, embed_dim)

sequence_length = 4 
embedding_dim = 8
query = generate_data(sequence_length, embedding_dim)
key = generate_data(sequence_length, embedding_dim)
value = generate_data(sequence_length, embedding_dim)
#compute attention scores
scores = np.dot(query, key.T) / np.sqrt(embedding_dim)
#apply softmax to get attention weights
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)
#compute the output as a weighted sum of the value vectors
output = np.dot(attention_weights, value)
print("Attention Weights:\n", attention_weights)
print("Output:\n", output)


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        Q = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final linear layer
        output = self.out_linear(context)
        
        return output, attention_weights


# Sample usage
seq_len, embed_dim = 4, 8
x = torch.rand(1, seq_len, embed_dim)  # Batch size of 1
attention_layer = MultiHeadAttention(embed_dim=embed_dim, num_heads=2)
output, attention_weights = attention_layer(x)
print("Attention Weights:\n", attention_weights)
print("Output:\n", output)