import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

#load a sample grayscale image

image = np.random.rand(10, 10) # Replace with actual image loading if needed

# Define convolutional filters 
edge_detection_filter = np.array([[ -1, -1, -1],
                                  [ -1,  8, -1],    
                                  [ -1, -1, -1]])
blur_filter = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])
sharpen_filter = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])
# Apply convolution with each filter
edge_detected_image = convolve(image, edge_detection_filter)
blurred_image = convolve(image, blur_filter)
sharpened_image = convolve(image, sharpen_filter)

# Visualize the original and filtered images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(edge_detected_image, cmap='gray')
axes[1].set_title('Edge Detected Image')
axes[1].axis('off')
axes[2].imshow(blurred_image, cmap='gray')
axes[2].set_title('Blurred Image')
axes[2].axis('off')
axes[3].imshow(sharpened_image, cmap='gray')
axes[3].set_title('Sharpened Image')
axes[3].axis('off')
plt.show()

import tensorflow as tf
#Create a simple input tensor
input_tensor = tf.random.normal([1, 10, 10, 1]) # Batch size of 1, 10x10 image, 1 channel
# Define a convolutional layer with the edge detection filter
conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')
#Apply convolution
output_tensor = conv_layer(input_tensor)
print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)

import torch
import torch.nn as nn

# Create a simple input tensor
input_tensor = torch.randn(1, 1, 10, 10) # Batch size of 1, 1 channel, 10x10 image
# Define a convolutional layer with the edge detection filter
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
# Apply convolution
output_tensor = conv_layer(input_tensor)
print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)


#tensorFlow Example
conv_layer_large_kernel = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same')
output_tensor_large_kernel = conv_layer_large_kernel(input_tensor)

print(f"Output Tensor Shape with Larger Kernel: {output_tensor_large_kernel.shape} (should be the same as input shape due to 'same' padding)")

#PyTorch Example
conv_layer_large_kernel_pytorch = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2) # padding=2 for 'same' padding with kernel size 5
conv_layer_stride_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
output_stride_2=conv_layer_stride_2(input_tensor)
print(f"Output Tensor Shape with Stride 2: {output_stride_2.shape} (should be smaller than input shape due to stride)") 