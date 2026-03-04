import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#DEfiine Transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize pixel values
])

#load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

#create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#Define a  CNN model in PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) # Input channels: 3 (RGB), Output channels: 6
        self.pool = nn.MaxPool2d(2, 2) # Max pooling layer with kernel size 2 and stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # Input channels: 6, Output channels: 16
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Fully connected layer (after pooling, image size is reduced to 5x5)
        self.fc2 = nn.Linear(120, 84) # Second fully connected layer
        self.fc3 = nn.Linear(84, 10) # Output layer for 10 classes
    def forward(self, x):
        x = F.relu(self.conv1(x)) # Apply first convolutional layer and ReLU activation
        x = self.pool(x) # Apply max pooling
        x = F.relu(self.conv2(x)) # Apply second convolutional layer and ReLU activation
        x = self.pool(x) # Apply max pooling
        x = x.view(-1, 16 * 5 * 5) # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x)) # Apply first fully connected layer and ReLU activation
        x = F.relu(self.fc2(x)) # Apply second fully connected layer and ReLU activation
        x = self.fc3(x) # Apply output layer
        return x
    
model= CNN()
print(model)

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate 0.001
#Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train() # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            running_loss += loss.item() # Accumulate loss
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}') # Print average loss for the epoch
train_model(model, train_loader, criterion, optimizer, num_epochs=10) # Train the model for 10 epochs
#Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs) # Forward pass  
            _, predicted = torch.max(outputs.data, 1) # Get the predicted class
            total += labels.size(0) # Update total count
            correct += (predicted == labels).sum().item() # Update correct count
    print(f'Test Accuracy: {100 * correct / total:.2f}%') #
evaluate_model(model, test_loader) # Evaluate the model on the test set

