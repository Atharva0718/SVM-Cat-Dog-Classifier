# SVM-Cat-Dog-Classifier

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs. The dataset is sourced from Kaggle: Dogs vs. Cats.
The goal is to train an SVM model to distinguish between cat and dog images based on pixel values and extracted features.
link:-https://www.kaggle.com/c/dogs-vs-cats/data

 Dataset
The dataset consists of images of cats and dogs in separate training and testing sets.
train/ - Contains labeled images of cats and dogs
test/ - Contains unlabeled images for prediction

Model Training Steps
Data Preprocessing

Resize images to a fixed size (e.g., 128x128 pixels)
Convert images to grayscale or extract features using HOG (Histogram of Oriented Gradients)
Normalize pixel values
Feature Extraction

Use HOG, PCA, or CNN features to reduce image dimensions for SVM
Training SVM Classifier

Train an SVM with an RBF or linear kernel on extracted features
Use GridSearchCV to find the best hyperparameters
Evaluation

Test accuracy on unseen images
Confusion matrix and classification report

Results
The SVM model achieved X% accuracy on the test dataset.
Feature extraction methods like HOG and PCA improved classification performance.
Future improvements can include CNN-based feature extraction.

