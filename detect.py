import os

from ultralytics import YOLO
import cv2

TRESHOLD = 0.5

cap = cv2.VideoCapture(0)
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
model = YOLO(model_path)

while(True):
    ret, frame = cap.read()
    H, W, _ = frame.shape

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, x2, y1, y2, score, class_id = result

        print(score)

        if score > TRESHOLD:
            v2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
