from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model

#Define a simplified Transformer ENcoder Block
def transformer_encoder(input_dim, num_heads, ff_dim):
    input = Input(shape=(None, input_dim))
    #Multi-head attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(input, input)
    attention_output = Add()([input, attention_output])  # Residual connection
    attention_output = LayerNormalization()(attention_output)
    #Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(input_dim)(ff_output)
    outputs= Add()([attention_output, ff_output])  # Residual connection
    outputs = LayerNormalization()(outputs)
    return Model(inputs=input, outputs=outputs)
# Create and visualize the model a sample transformer model 
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file='transformer_encoder.png')


