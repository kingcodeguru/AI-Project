import cv2                      # Camera library
import tensorflow as tf         # Deep learning library
from keras import models
import numpy as np              # Math library
import matplotlib.pyplot as plt # Displaying graphs library
import os
import pygame
from io import BytesIO

import constants

class_names = constants.class_names
MODEL_PATH = constants.MODEL_PATH
model = models.load_model(MODEL_PATH)
print("Model loaded successfully.")
IMAGE_SIZE = constants.IMAGE_SIZE

# the square we are taking of the camera
sq_start = constants.SQUARE_START
sq_size = constants.SQUARE_SIZE
sq_end = constants.SQUARE_START + constants.SQUARE_SIZE

def prepare_image(camera_img):
    cropped = camera_img[sq_start:sq_end, sq_start:sq_end]
    resized = (cv2.cvtColor(cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)) / 255.0
    data = resized.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    return data
def predict_image(ready_img):
    return model.predict([ready_img], verbose=0)[0]
def to_category(class_index):
    return class_names[class_index]


def load_and_preprocess(img_path, img_size=(64, 64), color=True):
    # Load image file as tensor
    img = tf.io.read_file(img_path)
    channels = 3 if color else 1
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale to [0,1]

    # Resize with padding to maintain aspect ratio
    img = tf.image.resize_with_pad(img, img_size[0], img_size[1])

    # Convert to NumPy array and add batch dimension
    img_array = np.expand_dims(img.numpy(), axis=0)
    return img_array

def predict_from_path(model, img_path, img_size=(64, 64), color=True, show=False):
    img_array = load_and_preprocess(img_path, img_size, color)
    preds = model.predict(img_array)

    # If using sparse_categorical_crossentropy → softmax output
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    print(f"Image: {img_path}")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    if show:
        plot_image(img_array[0], preds[0], confidence, color=color)
    return predicted_class, confidence

def plot_image(img, preds, confidence, color=True):
    _, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    class_name = class_names[np.argmax(preds)]
    color = None if color else 'gray'
    axes[0].imshow(img, cmap=color)
    axes[0].set_title(f"Predicted: {class_name}\nConfidence: {confidence:.4f}")
    axes[0].axis('off')
    axes[1].bar(class_names, preds)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust()
    plt.show()


last_update = 0
def plot_bar(preds, time):
    global last_update
    if time - last_update >= 0:
        last_update = time
        plt.clf()
        fig, ax = plt.subplots(dpi=100, figsize=(5, 4))
        fig.subplots_adjust(bottom=0.2)
        ax.bar(class_names, preds)
        ax.set_ylim(0, 1.0)
        ax.set_title('Confidence Graph')
        ax.set_xticklabels(class_names, rotation=45, ha='right')

        buf = BytesIO()
        fig.savefig(buf, format="PNG")
        buf.seek(0)

        graph = pygame.image.load(buf)
        return graph
    return None

if __name__ == '__main__':
    directory_path = "/home/liel/PyCharmMiscProject/data/Fruit262/archive/our_fruits4/date/1000.jpg"
    choice = 'y'
    if choice == 'y':
        choice = directory_path
        while choice != 'n':
            predict_from_path(model, choice, img_size=(64, 64), show=True)
            choice = input('next image path: ')
    else:
        success = []
        fail = []
        for item_name in os.listdir(directory_path):
            img_path = os.path.join(directory_path, item_name)
            if os.path.isfile(img_path):
                a, c = predict_from_path(model, img_path, img_size=(28, 28), color=False)
                name = f'{chr(a + ord('A'))}{item_name[0]}'
                if name[0] == name[1]:
                    success.append((name, c))
                else:
                    fail.append((name, c))
        print(f'Succeed {len(success)}/{len(success)+len(fail)} with average confidence {sum(t for _,t in success) / len(success):.4}: {success}')
        print(f'Failed  {len(fail)}/{len(success)+len(fail)} with average confidence {sum(t for _,t in fail) / len(fail):.4}: {fail}')
