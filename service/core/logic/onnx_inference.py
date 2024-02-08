import cv2
import numpy as np
from PIL import Image
from collections import Counter
import time
import main as main
import psutil
from service.core.logic.system_monitor import monitor_memory_usage

cpu_usage = psutil.cpu_percent(interval=1)  # Check every 1 s



def emotions_detector(video):
    emotion_list = []

    while True:
        ret, frame = video.read()  # Is 'image' numpy array?
        if not ret:
            break

        pil_image = Image.fromarray(frame)
        image = np.array(pil_image)

        time_init = time.time()  # time started for calculating inference time

        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces_detected = face_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces in the image

        for (x, y, w, h) in faces_detected:
            # Crop and resize the face region for the model/preprocessing
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (256, 256))
            im = np.float32(face)
            im = np.expand_dims(im, axis=0)

            time_elapsed_preprocessing = f'{time.time() - time_init:.4f} secs'

            # Inference by passing the face to the model
            print(f"CPU Usage 4: {cpu_usage}%")
            monitor_memory_usage()

            onnx_pred = main.model.run(['dense'], {'input_image': im})
            print(f"CPU Usage 5: {cpu_usage}%")
            monitor_memory_usage()

            # Predict emotion
            class_names = ['angry', 'happy', 'neutral', 'sad', 'surprised']
            emotion = class_names[np.argmax(onnx_pred[0][0])]

            time_elapsed = f'{time.time() - time_init:.4f} secs'

            emotion_list.append(emotion)

    return emotion_list


def emotion_detection_result(p_emotion_list):
    len_emotion_list = len(p_emotion_list)
    element_counts = Counter(p_emotion_list)
    result_emotions = []
    threshold = 0.3
    for e, cnt in element_counts.items():
        if e == "neutral":
            continue
        else:
            percent = cnt / len_emotion_list
            if percent >= threshold:
                result_emotions.append(e)

    if len(result_emotions) == 0:
        result_emotions.append('neutral')

    return {'emotion': result_emotions,  ####
            'time_elapsed_preprocessing': str(time_elapsed_preprocessing),
            'time_elapsed': str(time_elapsed)
            }
