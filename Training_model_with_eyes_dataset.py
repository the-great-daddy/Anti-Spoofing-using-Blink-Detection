# TRAINING THE CNN MODEL TO DETECT THE CLOSED OR OPENED EYES.
# AFTER TRAINING WE SAVE THE MODEL FOR FUTURE USE.

import os

import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix

IMG_SIZE = 24


def collect():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        horizontal_flip=True,
    )

    train_gen = train_datagen.flow_from_directory(
        directory="dataset/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        directory="dataset/val",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )
    return train_gen, val_gen


def save_model(model):
    model.save("eye_status_classifier.h5")


def load_pretrained_model():
    model = load_model("eye_status_classifier.h5")
    model.summary()
    return model


def train(train_generator, val_generator):
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

    print("[LOG] Initialize Convolutional Neural Network...")

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=120, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=84, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation="sigmoid"))

    print(model.summary())

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=2,
    )
    return model
    # save_model(model)


def predict(img, model):
    img = Image.fromarray(img, "RGB").convert("L")
    img = np.array(img)
    img = resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    img /= 255
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = "closed"
    elif prediction > 0.90:
        prediction = "open"
    else:
        prediction = "idk"
    return prediction


def evaluate(x_test, y_test):
    model = load_model()
    print("Evaluate model")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(acc * 100)


TRAIN_GENERATOR, VAL_GENERATOR = collect()

model = train(TRAIN_GENERATOR, VAL_GENERATOR)


# Taking images data from directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath)
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)


# Example usage
image_directory = "dataset/val/closed"
closed_images = load_images_from_directory(image_directory)
open_images = load_images_from_directory('dataset/val/open')

print(open_images.shape)
# test_images = np.array(test_images)

predicted_open = np.array(model.predict(open_images))
predicted_close = np.array(model.predict(closed_images))

# predicted_images = np.concatenate(predicted_open, predicted_close)

print(predicted_open.shape)

test_images = np.ones(542)
test_images = test_images.reshape(-1, 1)
print(test_images.shape)
print(accuracy_score(test_images, predicted_open))
print(confusion_matrix(test_images, predicted_open))

# take all images from test/open , test/close
# get the prediction of the image and store it in prediction array
# make y_label array
# print out accuracy_score and confusion_matrix
