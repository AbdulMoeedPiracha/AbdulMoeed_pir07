import os
import numpy as np
import cv2  # OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_images(folder):
    images = []
    labels = []
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    
   
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        
        if 'cat' in file:
            label = 'cat'
        elif 'dog' in file:
            label = 'dog'
        else:
            print(f"Unrecognized file: {file}")
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img is not None:
            img = cv2.resize(img, (64, 64)) 
            images.append(img.flatten())  
            labels.append(label)
        else:
            print(f"Failed to load image: {img_path}")

    return np.array(images), np.array(labels)


dataset_path = 'D:\\train\\train'


X, y = load_images(dataset_path)


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


svm = SVC(kernel='linear')  
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
