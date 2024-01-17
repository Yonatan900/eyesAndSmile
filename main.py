import cv2
import os
import random


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# Load images from '159.people' folder
images = load_images_from_folder('159.people')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_images = []
smile_images = []
no_smile_eyes = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.05, 1)

    if len(eyes) > 0:
        eye_images.append(img)

    # Detect smiles
    smiles = smile_cascade.detectMultiScale(gray, 1.05, 1)

    if len(smiles) > 0:
        smile_images.append(img)

    # no eyes and smiles
    if len(smiles) == 0 and len(eyes) == 0:
        no_smile_eyes.append(img)

# Select 5 random images
print(len(images))
print(len(no_smile_eyes))
random_images = random.sample(smile_images, 5)
for i, img in enumerate(random_images):
    cv2.imshow(f'Image {i}', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
