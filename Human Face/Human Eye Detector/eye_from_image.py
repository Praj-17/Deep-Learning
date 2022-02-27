import cv2
# Load the Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

# Define function that will do detection
def detect_eye(gray, color):
  """ Input = greyscale image or frame from video stream
      Output = Image with rectangle box in the face
  """
  # Now get the tuples that detect the faces using above cascade

  # faces are the tuples of 4 numbers
  # x,y => upperleft corner coordinates of face
  # width(w) of rectangle in the face
  # height(h) of rectangle in the face
  # grey means the input image to the detector
  # 1.3 is the kernel size or size of image reduced when applying the detection
  # 5 is the number of neighbors after which we accept that is a face
  
  # Now iterate over the faces and detect eyes
  
    
  # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
  # Detect eyes now
  eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
  if len(eyes) == 0:
      print("No eyes detected")
  # Now draw rectangle over the eyes
  else:
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
  return color
    
# Capture video from webcam 
frame = cv2.imread("eye3.jpeg")
# Run the infinite loop 

# Read each frame
# Convert frame to grey because cascading only works with greyscale image
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Call the detect function with grey image and colored frame
canvas = detect_eye(gray, frame)
# Show the image in the screen
cv2.imshow("Video", canvas)
# Put the condition which triggers the end of program
cv2.waitKey(0)
cv2.destroyAllWindows()