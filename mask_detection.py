import cv2


face_haarcascade = './trained-classifiers/haarcascade_frontalface_alt2.xml'
mouth_haarcascade = './trained-classifiers/haarcascade_mouth.xml'

faceClassifier = cv2.CascadeClassifier(face_haarcascade)
mouthClassifier = cv2.CascadeClassifier(mouth_haarcascade)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_COUNT, 15)


while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()

    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x,y), (x + w, y + h), (0,255,0), 2)

        mouth_rects = mouthClassifier.detectMultiScale(gray, 1.5, 5)

        for (mx, my, mw, mh) in mouth_rects:
            if(y < my < y + h):
                cv2.rectangle(frame_color, (x,y), (x + w, y + h), (0,0,255), 2)
                cv2.rectangle(frame_color, (mx, my), (mx + mh, my + mw),  (255,0,0), 5)
                break
    cv2.imshow('Preview', frame_color)

capture.release()
cv2.destroyAllWindows()