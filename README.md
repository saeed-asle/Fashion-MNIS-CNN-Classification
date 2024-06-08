# Fashion MNIST Classification using Convolutional Neural Networks (CNN)
Authored by saeed asle
# Description
This project demonstrates the use of Convolutional Neural Networks (CNNs) for classifying images in the Fashion MNIST dataset.
The Fashion MNIST dataset consists of grayscale images of clothing items belonging to 10 categories.

# steps:
* Data Loading and Preprocessing: Loads the Fashion MNIST dataset and preprocesses the images for training and testing.
* Model Creation: Defines a CNN model using Keras with convolutional and pooling layers followed by dense layers.
* Model Compilation and Training: Compiles the model with an initial learning rate and a learning rate scheduler, and trains it on the training data with data augmentation.
* Model Evaluation: Evaluates the trained model on the test data and prints the classification report.
* Model Saving and Loading: Saves the trained model to a file and loads it back for inference.
* Inference: Uses the trained model to make predictions on a sample of test images and displays the images with their true labels and predicted labels.
# Features
* Data Loading and Preprocessing: Loads images from the Fashion MNIST dataset and preprocesses them for training and testing.
* CNN Model Creation: Defines a CNN model using Keras with convolutional, pooling, and dense layers.
* Model Compilation and Training: Compiles the model with an initial learning rate and a learning rate scheduler, and trains it on the training data with data augmentation.
* Model Evaluation: Evaluates the trained model on the test data and prints the classification report.
* Model Saving and Loading: Saves the trained model to a file and loads it back for inference.
* Inference: Uses the trained model to make predictions on a sample of test images and displays the images with their true labels and predicted labels.
# Dependencies
* sklearn.metrics: For calculating the classification report.
* keras.optimizers.Adam: For using the Adam optimizer in the model compilation.
* keras.callbacks.LearningRateScheduler: For implementing the learning rate scheduler during training.
* keras.layers.Flatten, keras.layers.Dense: For defining the fully connected layers in the model.
* keras.layers.Conv2D, keras.layers.MaxPooling2D: For defining the convolutional and pooling layers in the model.
* keras.models.Sequential: For defining the sequential model architecture.
* keras.preprocessing.image.ImageDataGenerator: For data augmentation.
* tensorflow, numpy, matplotlib.pyplot: For general operations and plotting.
# How to Use
* Ensure you have the necessary libraries installed, such as sklearn, keras, tensorflow, numpy, and matplotlib.
* Load the Fashion MNIST dataset using tf.keras.datasets.fashion_mnist.
* Preprocess the data by normalizing the pixel values to the range [0, 1].
* Define a CNN model using keras.models.Sequential with convolutional, pooling, and dense layers.
* Compile the model with an initial learning rate and a learning rate scheduler.
* Train the model on the training data using data augmentation with keras.preprocessing.image.ImageDataGenerator.
* Evaluate the model on the test data and print the classification report.
* Save the trained model to a file using model.save and load it back using tf.keras.models.load_model for inference.
* Use the trained model to make predictions on a sample of test images and display the images with their true and predicted labels.
# Output
* The code outputs various results and visualizations, including:
* Training and test data shape and label information.
* Sample images from the dataset with their corresponding labels.
* Model summary and details.
* Training progress and metrics (loss, accuracy).
* Test accuracy and classification report.
* Saved and loaded model for inference.
* Sample test images with their true and predicted labels.
