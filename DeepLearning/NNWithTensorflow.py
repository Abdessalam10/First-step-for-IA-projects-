import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28,28,1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28,28,1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

#build the model 
model = Sequential([
    # Number of kernels, filter size scanning the image in 3×3 patches,activation function,input_shape=(28,28,1): Expected input dimensions - 28×28 pixels with 1 channel (grayscale)
    Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)), #a mathematical operation called convolution where the filter multiplies its values with corresponding pixels
    MaxPooling2D((2,2)), #reduces the spatial dimensions of the feature maps by taking the maximum value in each 2×2 block
    Flatten(), #converts the 2D feature maps into a 1D vector, preparing the data for the fully connected layers
    Dense(128,activation="relu"), #fully connected layer with 128 neurons and ReLU activation function, which helps the model learn complex patterns in the data
    Dropout(0.5), #a regularization technique that randomly sets 50% of the input units to zero during training, which helps prevent overfitting by forcing the model to learn more robust features
    Dense(10,activation="softmax") #output layer with 10 neurons (one for each class) and softmax activation function, which converts the output into probabilities for each class
])
#Display the model architecture
model.summary()

#Compile the model
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#Train the model
                    #Training dat,Complete passes (see dataset 10 times),Mini-batch training: Processes 32 images,Validation split: 20% of the training data is used for validation during training, allowing us to monitor the model's performance on unseen data and prevent overfitting.
history= model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

#Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

model.save("mnist_classifier.h5")

#load the model
from tensorflow.keras.models import load_model
loaded_model = load_model("mnist_classifier.h5")

#verify the loaded model's performance
loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)  
print(f"Loaded model test accuracy: {loaded_test_accuracy:.4f}")
