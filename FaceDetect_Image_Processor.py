import cv2

# Loading the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loading the image
img = cv2.imread("FacesPic.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting faces in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)

# Drawing rectangles around detected faces
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Displaying the result
resized_img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
cv2.imshow("Detected Faces", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
