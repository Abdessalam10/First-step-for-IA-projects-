import tensorflow as tf
from tensorflow.keras import models,layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Load CIFAR-10 dataset  
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15, # Randomly rotate images by 15 degrees
    width_shift_range=0.1, # Randomly shift images horizontally by 10% of the width
    height_shift_range=0.1, # Randomly shift images vertically by 10%
    horizontal_flip=True, # Randomly flip images horizontally
)

#Fit the data generator to the training data
datagen.fit(x_train)

def create_model():
    model = models.Sequential()
    # Convolutional layers with batch normalization and dropout for regularization
    model.add(layers.Input(shape=(32,32,3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu')) 
    model.add(layers.BatchNormalization()) # Batch normalization
    model.add(layers.Conv2D(32, (3, 3), activation='relu')) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25)) # Dropout for regularization
    
    # Add more convolutional layers with batch normalization and dropout
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.BatchNormalization()) # Batch normalization
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25)) # Dropout for regularization
    
    #Fully connected layers with L2 regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))) # L2 regularization
    model.add(layers.BatchNormalization()) # Batch normalization
    model.add(layers.Dropout(0.5)) # Dropout for regularization
    model.add(layers.Dense(10, activation='softmax')) # Output layer for
    
    
    
    return model

model= create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]//64) # Train the model with data augmentation

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # Evaluate the model on the test set
print(f'Test accuracy: {test_acc:.4f}')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')  
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt