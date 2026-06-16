import cv2
from engine.Object_detection import ObjectDetection


cap = cv2.VideoCapture("pytorch\data\crowd_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break





    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()