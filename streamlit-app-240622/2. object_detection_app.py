import streamlit as st
import numpy as np
import cv2
from PIL import Image


def main():

    def process_image(image):
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 0.007843, (300, 300), 127.5)
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        net.setInput(blob=blob)
        detections = net.forward()
        return detections

    def annotate_image(image, detections, confidence_threshold=0.5):
        (h, w) = image.shape[:2]
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')
                cv2.rectangle(image, (start_x, start_y),
                              (end_x, end_y), 70, 2)
        return image

    MODEL = '/Users/dwctran/AIO-Exercise/streamlit-app-240622/model/MobileNetSSD_deploy.caffemodel'
    PROTOTXT = '/Users/dwctran/AIO-Exercise/streamlit-app-240622/model/MobileNetSSD_deploy.prototxt.txt'

    st.write('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption='Uploaded Image')

        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image=image)
        processed_image = annotate_image(image=image, detections=detections)
        st.image(processed_image, caption="Processed Image")


if __name__ == '__main__':
    main()
