import cv2
import pickle
import json
import numpy as np
import base64
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}


    global __model
    if __model is None:
        with open("./artifacts/saved_classifier_model.pkl","rb") as f:
            __model = pickle.load(f)

def class_number_to_name(number):
    return __class_number_to_name[number]


def classifyImage(image_based_64, file_path = None):
    imgs = getCroppedImageWith2Eyes( file_path, image_based_64)

    result = []

    for img in imgs:
        scaled_raw = cv2.resize(img, (32, 32))
        img_transformed = w2d(img, 'db1', 5)
        scaled_transformed = cv2.resize(img_transformed, (32, 32))
        combined_img = np.vstack((scaled_raw.reshape(32 * 32 * 3, 1), scaled_transformed.reshape(32 * 32, 1)))

        len_image_combined = 32 *32 * 3 + 32* 32

        final = combined_img.reshape(1, len_image_combined).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number

        })

    return result


def getCroppedImageWith2Eyes(image_path, image_based_64):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64(image_based_64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    croppped_faces = []
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            croppped_faces.append(roi_color)
    return croppped_faces



def get_cv2_image_from_base64(image_base_64):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = image_base_64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_based_64_for_test():
    with open("base64.txt") as f:
        return f.read()


if  __name__ == "__main__":
    load_saved_artifacts()
    print(classifyImage(get_based_64_for_test(), None))

