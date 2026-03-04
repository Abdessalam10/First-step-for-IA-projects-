import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
#load dataset
transform=transforms.ToTensor() # Convert images to PyTorch tensors
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convert PyTorch datasets to numpy arrays for TensorFlow
def pytorch_to_numpy(dataset):
    data = []
    labels = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        # Convert from PyTorch tensor (C, H, W) to numpy array (H, W, C)
        image_np = image.permute(1, 2, 0).numpy()
        data.append(image_np)
        labels.append(label)
    return np.array(data), np.array(labels)

# Convert datasets for TensorFlow use
X_train, y_train = pytorch_to_numpy(train_dataset)
X_test, y_test = pytorch_to_numpy(test_dataset)

#visualize sample images

fig,axes = plt.subplots(1,5, figsize=(15,3))
for i in range(5):
    image, label = train_dataset[i]
    axes[i].imshow(image.permute(1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')
plt.show()
#Display pexel values of the first image
image, label = train_dataset[0]
print("Pixel values of the first image:")
print(image)    
print("Label of the first image:", label)

import tensorflow as tf

#Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)   

import torch.nn as nn
import torch.nn.functional as F
#Define a simple CNN model in PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 15 * 15) # Flatten the output from the convolutional layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    