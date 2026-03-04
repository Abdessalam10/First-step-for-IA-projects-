import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter,uniform_filter

#Create a sample feature map
feature_map = np.array([[1, 2, 3, 0],
                        [0, 1, 2, 3],   
                        [3, 0, 1, 2],
                        [2, 3, 0, 1]])

#max pooling (2x2 kernel, stride 2  )
pooled_feature_map = maximum_filter(feature_map, size=(2,2), mode='constant', cval=0)
print("Original Feature Map:")
print(feature_map)
print("\nPooled Feature Map (2x2 max pooling):")
print(pooled_feature_map)
#average pooling (2x2 kernel, stride 2)
pooled_feature_map_avg = uniform_filter(feature_map, size=(2,2), mode='constant', cval=0)
print("\nPooled Feature Map (2x2 average pooling):")
print(pooled_feature_map_avg)

#Plot
fig , axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(feature_map, cmap='viridis')
axes[0].set_title("Original Feature Map")
axes[1].imshow(pooled_feature_map, cmap='viridis')
axes[1].set_title("Max Pooled Feature Map")
axes[2].imshow(pooled_feature_map_avg, cmap='viridis')
axes[2].set_title("Average Pooled Feature Map")
plt.tight_layout()
plt.show()

import tensorflow as tf
#Create a simple input tensor
input_tensor = tf.constant(feature_map.reshape(1, 4, 4, 1),dtype=tf.float32) # Batch size of 1, 4x4 image, 1 channel
# Define a max pooling layer
max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' )
# Apply max pooling
pooled_tensor = max_pool_layer(input_tensor)
print("Input Tensor Shape:", input_tensor.shape)
print("Pooled Tensor Shape:", pooled_tensor.shape)
#Avg Pooling
avg_pool_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2 ), strides=2, padding='valid')    
pooled_tensor_avg = avg_pool_layer(input_tensor)
print("Pooled Tensor Shape (Average Pooling):", pooled_tensor_avg.shape)
print ("\nPooled Tensor (Average Pooling):\n", pooled_tensor_avg.numpy())   

print("\n\n\n")
import torch
import torch.nn as nn
# Create a simple input tensor
input_tensor = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Batch size of 1, 1 channel, 4x4 image 
# Define a max pooling layer
max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
# Apply max pooling
pooled_tensor = max_pool_layer(input_tensor)
print("max pooled tensor:", pooled_tensor.squeeze().numpy()    )
#Average Pooling
avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
pooled_tensor_avg = avg_pool_layer(input_tensor)
print("Average Pooled Tensor Shape:", pooled_tensor_avg.shape)
print("\nPooled Tensor (Average Pooling):\n", pooled_tensor_avg.squeeze().numpy())  

#Tensorflow Example
model_tf = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Flatten(),
])

#PyTorch Example
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x =  self.pool2(x)
        return x