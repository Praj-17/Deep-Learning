import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")





frame = cv2.imread("Images\\demo.jpeg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

if len(faces) != 0:
    
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),3)
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        cv2.putText(frame,str("Face"),(x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # for point in landmarks.parts():
        #     cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), 3)
else:
    print("No face detected")
cv2.imshow("My Screen", frame)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
    


