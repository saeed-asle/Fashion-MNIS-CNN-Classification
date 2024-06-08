from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("Training data shape:", train_images.shape)
print("Number of training labels:", len(train_labels))
print("Training labels:", train_labels)
print("Test data shape:", test_images.shape)
print("Number of test labels:", len(test_labels))
# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display some sample images
plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)  # Squeeze to remove the channel dimension
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Define the model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input

model = Sequential([
    Conv2D(32, (3, 3), activation='tanh', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model with an initial learning rate and learning rate scheduler
initial_learning_rate = 0.001

def lr_schedule(epoch):
    return initial_learning_rate * 0.2**int(epoch / 10)

lr_scheduler = LearningRateScheduler(lr_schedule)
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
history = model.fit(train_images, train_labels, batch_size=32, epochs=3,callbacks=[lr_scheduler])

# Evaluate the model and print classification report
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Use model.predict to get class probabilities for each sample
predicted_probabilities = model.predict(test_images)

# Get the class with the highest probability for each sample
predicted_classes = np.argmax(predicted_probabilities, axis=1)

# Generate the classification report
from sklearn.metrics import classification_report
print(classification_report(test_labels, predicted_classes, target_names=class_names))

# Save and load the model
model.save("fashion_mnist_model.h5")
loaded_model = tf.keras.models.load_model("fashion_mnist_model.h5")

num_images_to_display = 10  # Number of images to display
sample_indices = np.random.choice(len(test_images), num_images_to_display)

for i, index in enumerate(sample_indices):
    image = test_images[index]
    true_label = test_labels[index]
    
    # Use the model to make predictions
    predicted_probabilities = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(predicted_probabilities)
    
    # Display the image with true label and predicted label
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image.squeeze(), cmap=plt.cm.binary)  # Squeeze to remove the channel dimension
    plt.xlabel(f'True: {class_names[true_label]}\nPredicted: {class_names[predicted_class]}')

plt.show()