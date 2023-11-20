import cv2
import datetime
import pandas as pd

# Initializing variables
first_frame = None  # To store the first frame for comparison
video_capture = cv2.VideoCapture(0)  # Accessing the webcam

timing = [None, None]  # List to store timing information
status_list = [None, None]  # List to track status changes
df = pd.DataFrame(columns=["Start", "End"])  # Creating an empty DataFrame to store motion time intervals

while True:
    check, frame = video_capture.read()  # Capturing video frame by frame
    status = 0

    # Converting the frame to grayscale and applying Gaussian blur for noise reduction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray  # Setting the first frame for reference
        continue

    # Calculating the difference between the current frame and the reference frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # Creating a threshold frame to highlight significant differences
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Smoothing the threshold frame to improve accuracy
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contours in the threshold frame to identify motion
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        # Drawing rectangles around moving objects (detected contours)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    status_list.append(status)

    # Tracking changes in motion status to capture the timing of motion events
    if status_list[-1] == 1 and status_list[-2] == 0:
        timing.append(datetime.datetime.now())  # Start time of motion

    if status_list[-1] == 0 and status_list[-2] == 1:
        timing.append(datetime.datetime.now())  # End time of motion

    # Displaying the frames
    cv2.imshow("RGB Frame", frame)
    cv2.imshow("Threshold Frame", thresh_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if status == 1:
            timing.append(datetime.datetime.now())  # Recording end time if motion persists at exit
        break

# Storing motion timing information in a DataFrame and saving it to a CSV file
for i in range(0, len(timing), 2):
    df = df.append({"Start": timing[i], "End": timing[i + 1]}, ignore_index=True)

df.to_csv("Motion_Times.csv")  # Saving motion timings to a CSV file

# Releasing the video capture and closing all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
