import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#Define Transformation
transform = transforms.Compose([
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to the range [-1, 1]
])

#Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    

print(f"Training data shape: {train_dataset.data.shape}, Training labels shape: {train_dataset.targets.shape}")
print(f"Testing data shape: {test_dataset.data.shape}, Testing labels shape: {test_dataset.targets.shape}") 
 
#Define the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()                      # Flatten layer to convert 2D images to 1D vectors
        self.fc1 = nn.Linear(28*28,128)      # Fully connected layer
        self.fc2 = nn.Linear(128, 6)               
        self.fc3 = nn.Linear(6, 10)              # Output layer for 10 classes
        

    def forward(self, x):
        x = self.flatten(x)                        # Flatten the input tensor
        x = F.relu(self.fc1(x))                   # Apply fully connected layer and ReLU activation
        x=F.relu(self.fc2(x))                   # Apply second fully connected layer and ReLU activation
        x = self.fc3(x)                           # Apply output layer
       
        return x          
    
model = NeuralNet()
print(model)
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
#train loop
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(images)  # Forward pass
            loss= criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            running_loss += loss.item()  # Accumulate the loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
train_model(model, train_loader, criterion, optimizer, epochs=10)
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Update total count
            correct += (predicted == labels).sum().item()  # Update correct count
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
evaluate_model(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), 'mnist_model.pth')
#Reload the model (if needed)
model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))
evaluate_model(model, test_loader)

#update optimizer with a new learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Update learning rate
train_model(model, train_loader, criterion, optimizer, epochs=5)  # Continue training
#evaluate the model again after further training
evaluate_model(model, test_loader)
