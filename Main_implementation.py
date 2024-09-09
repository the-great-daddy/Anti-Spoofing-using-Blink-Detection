# IMPLEMENTATION OF BLINK DETECTION...
# We are extracting face & eyes using Viola-Jones algo with pretrained HAAR Classifier and
# Checking the Blink using CNN

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from skimage.transform import resize

IMG_SIZE = 24


def load_pretrained_model():
    model = load_model("eye_status_classifier.h5")
    model.summary()
    return model


trained_model = load_pretrained_model()


def predict(img, model):
    img = Image.fromarray(img, "RGB").convert("L")
    img = np.array(img)
    resized_img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE)).astype("float32")
    resized_img /= 255
    image = resized_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(image)
    if prediction <= 0.5:
        prediction = "closed"
    else:
        prediction = "open"
    return prediction


def init():
    face_Detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    open_eyes_Detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    left_eye_Detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_Detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    # return (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, images)
    # return (face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, images)
    return face_Detector, open_eyes_Detector, left_eye_Detector, right_eye_Detector


def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status
         where a '0' means that the eyes were closed and '1' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0' * (i + 1) + '1'
        if pattern in history:
            return True
    return False


def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector,
                       eyes_detected):
    ret, frame = video_capture.read()

    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # For each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        gray_face = gray[y:y + h, x:x + w]

        # Eyes detection
        # First check if eyes are open (with glasses taken into account)
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # If open_eyes_glasses detect eyes, they are open
        if len(open_eyes_glasses) == 2:
            eyes_detected += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        else:
            # Separate the face into left and right sides
            left_face = face[:, int(w / 2):]
            left_face_gray = gray_face[:, int(w / 2):]

            right_face = face[:, :int(w / 2)]
            right_face_gray = gray_face[:, :int(w / 2)]

            eye_status = '1'  # Assume the eyes are open

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Check the left and right eyes
            for (ex, ey, ew, eh) in right_eye:
                pred = predict(right_face[ey:ey + eh, ex:ex + ew], model)
                color = (0, 0, 255) if pred == 'closed' else (0, 255, 0)
                if pred == 'closed':
                    eye_status = '0'
                cv2.rectangle(right_face, (ex, ey), (ex + ew, ey + eh), color, 2)

            for (ex, ey, ew, eh) in left_eye:
                pred = predict(left_face[ey:ey + eh, ex:ex + ew], model)
                color = (0, 0, 255) if pred == 'closed' else (0, 255, 0)
                if pred == 'closed':
                    eye_status = '0'
                cv2.rectangle(left_face, (ex, ey), (ex + ew, ey + eh), color, 2)
            eyes_detected += eye_status
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Now check if the person is blinking
        if isBlinking(eyes_detected, 4):
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, 'Real', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # if len(eyes_detected) > 20:  # Use a more specific condition if needed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, 'Fake', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return eyes_detected, frame


(face_detector, open_eyes_detector, left_eye_detector, right_eye_detector) = init()

print("[LOG] Opening webcam ...")

eyes_detected = ""

video_capture = cv2.VideoCapture(0)

while True:
    eyes_detected, frame = detect_and_display(trained_model, video_capture, face_detector, open_eyes_detector,
                                              left_eye_detector,
                                              right_eye_detector, eyes_detected)
    if len(eyes_detected) > 50:
        eyes_detected = ""
    cv2.imshow("Eye-Blink based Liveness Detection for Facial Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
