import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def load_images_and_labels_from_folder(folder):
    raw_images = []
    raw_labels = []
    for filename in os.listdir(folder):
        raw_img = cv2.imread(os.path.join(folder, filename))
        if raw_img is not None:
            raw_images.append(raw_img)
            label = int(filename[0])  # the first char represents the labels 0- none 1- eyes 2- smiles
            raw_labels.append(label)
    return raw_images, raw_labels


# Load images and labels
images, labels = load_images_and_labels_from_folder('159.people')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

predicted_labels_smiles = []
predicted_labels_eyes = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    label_smile = 0
    label_eye = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 1)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles within face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # found smile and label it
        if len(smiles) > 0:
            label_smile = 1
        # found eye and label it
        if len(eyes) > 0:
            label_eye = 1

    predicted_labels_smiles.append(label_smile)
    predicted_labels_eyes.append(label_eye)

# Replace non-relevant labels with 0 for categories 0 and 1
labels_eyes = [1 if label == 2 else label for label in labels]

# Replace non-relevant labels with 0 for categories 0 and 2
labels_smile = [1 if label == 2 else 0 if label == 1 else label for label in labels]

# generate the confusion matrixes
cm_eyes = confusion_matrix(labels_eyes, predicted_labels_eyes)
cm_smile = confusion_matrix(labels_smile, predicted_labels_smiles)

# Calculate accuracy for eyes detection
accuracy_eyes = accuracy_score(labels_eyes, predicted_labels_eyes)

# Calculate accuracy for smiles detection
accuracy_smile = accuracy_score(labels_smile, predicted_labels_smiles)

# Create a heatmap for cm_eyes
plt.figure(figsize=(10, 7))
sns.heatmap(cm_eyes, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Eyes')
plt.text(0, -0.7, f'Accuracy for Eyes Detection: {accuracy_eyes * 100:.2f}%', fontsize=12)
plt.show()

# Create a heatmap for cm_smile
plt.figure(figsize=(10, 7))
sns.heatmap(cm_smile, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Smiles')
plt.text(0, -0.7, f'Accuracy for Smiles Detection: {accuracy_smile * 100:.2f}%', fontsize=12)
plt.show()
