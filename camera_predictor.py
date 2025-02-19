import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import hog, local_binary_pattern

# Load the trained model and scaler
model = joblib.load("improved_svm_classifier.pkl")
scaler = joblib.load("scaler.pkl")

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

# Function to preprocess the image
def preprocess_image(img, img_size=(64, 64)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, img_size)  # Resize to 64x64
    img = cv2.equalizeHist(img)  # Improve contrast
    return img

# Function to predict cat or dog
def predict_image(img):
    # Preprocess the image
    img_processed = preprocess_image(img)
    
    # Extract features
    features = extract_features(img_processed)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)
    return "Cat" if prediction[0] == 0 else "Dog"

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Predict the image
    prediction = predict_image(frame)
    
    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Camera - Cat vs Dog Classifier", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()