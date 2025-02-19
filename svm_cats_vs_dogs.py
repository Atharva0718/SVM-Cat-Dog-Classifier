import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from skimage.feature import hog, local_binary_pattern
from skimage.transform import rotate
from skimage.util import random_noise
from imblearn.over_sampling import SMOTE

# Define the base path to your dataset
base_path = r"D:\Internship of prodigy\Task 3\cats_vs_dogs"

# Function to augment images
def augment_image(img):
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Horizontal flip
    augmented_images.append(cv2.flip(img, 1))
    
    # Rotation (e.g., 15 degrees)
    augmented_images.append(rotate(img, angle=15, mode='reflect'))
    
    # Add noise
    augmented_images.append((random_noise(img, mode='gaussian', var=0.01) * 255).astype(np.uint8))
    
    # Brightness adjustment
    augmented_images.append(np.clip(img * 1.2, 0, 255).astype(np.uint8))  # Increase brightness
    
    return augmented_images

# Function to extract LBP features
def extract_lbp_features(img, P=8, R=1):
    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

# Function to extract combined features (HOG + LBP)
def extract_features(img):
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
    lbp_features = extract_lbp_features(img)
    return np.hstack([hog_features, lbp_features])

# Function to load and preprocess images
def load_image(img_path, label, img_size=(64, 64)):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return [], []
        img = cv2.resize(img, img_size)
        img = cv2.equalizeHist(img)  # Improve contrast
        
        # Generate augmented images
        augmented_images = augment_image(img)
        
        features = []
        for im in augmented_images:
            fd = extract_features(im)  # Use combined features
            features.append(fd)
        
        return features, [label]*len(features)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return [], []

# Function to load images using multithreading
def load_images_from_folder(folder, label, img_size=(64, 64)):
    images, labels = [], []
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return np.array(images), np.array(labels)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, os.path.join(folder, file), label, img_size) 
                   for file in os.listdir(folder)]
        for future in futures:
            feats, lbls = future.result()
            if feats:
                images.extend(feats)
                labels.extend(lbls)
    
    return np.array(images), np.array(labels)

# Load dataset
cats_folder = os.path.join(base_path, "train/cats")
dogs_folder = os.path.join(base_path, "train/dogs")

cat_images, cat_labels = load_images_from_folder(cats_folder, label=0)
dog_images, dog_labels = load_images_from_folder(dogs_folder, label=1)

# Verify data loading
if len(cat_images) == 0 or len(dog_images) == 0:
    print("Error: No images loaded. Check paths and loader.")
    exit()

# Combine and shuffle data
X = np.vstack((cat_images, dog_images))
y = np.hstack((cat_labels, dog_labels))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = [
    {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['poly'], 'degree': [2, 3]},
    {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['sigmoid']}
]

svm = SVC(random_state=42)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train_res, y_train_res)

best_clf = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

# Evaluate
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
joblib.dump(best_clf, "improved_svm_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

# Generate submission
submission_df = pd.DataFrame({'ImageId': range(1, len(y_pred)+1), 'Label': y_pred})
submission_df.to_csv("improved_submission.csv", index=False)
print("Submission file created. ✅")