import os
import random

import cv2
from ultralytics import YOLO


TRESHOLD = 0.8

model = YOLO("cards.pt") # model file
cap = cv2.VideoCapture(0)

colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for name in model.names
] # generate random colors for each class

while True:
    ret, frame = cap.read() # read webcam input
    frame = cv2.flip(frame, 1) # flip horizontal
    results = model(frame)[0] # inference

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > TRESHOLD:
            # only draw bounding boxes when we meet our treshold
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(class_id)], 3
            ) # draw the bounding box
            cv2.putText(
                frame,
                results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                colors[int(class_id)],
                3,
                cv2.LINE_AA,
            ) # draw the class name

    # display the resulting frame
    cv2.imshow("Object detection with YOLOv8", frame)

    # break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
