# python test.py --video sheep_running_2.mp4 --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
# Detection
import numpy as np
import argparse
import cv2
import sys
from math import cos, sin, sqrt

# Tracking
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

def detect(frame):
    cv2.imwrite("first_frame.jpg",frame)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation). Also resize original image to correspond to tracker
    image = cv2.imread("first_frame.jpg")
    image = imutils.resize(image, width=500)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    locations = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]

    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
    	if confidence > args["confidence"]:
    		# extract the index of the class label from the `detections`,
    		# then compute the (x, y)-coordinates of the bounding box for
    		# the object
    		idx = int(detections[0, 0, i, 1])
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    		locations.append([startX, startY, endX, endY])
    		#print(location)

    		# display the prediction
    		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
    		print("[INFO] {}".format(label))
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			COLORS[idx], 2)
    		y = startY - 15 if startY - 15 > 15 else startY + 15
    		cv2.putText(image, label, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return locations, image


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Captures first frame of the video, and write to an image for initial detection
vs = cv2.VideoCapture(args["video"])
success,first_frame = vs.read()
#cv2.imwrite("first_frame.jpg",first_frame)

locations, image = detect(first_frame)
# show the output image
cv2.imshow("first_frame", image)
cv2.waitKey(0)

# Define parameters for Lucas Kanade Optical Flow method
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Define paramaters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
#p0 = np.array([])
p0 = np.array([[]])

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	#print(trackers)

# initialize the bounding box coordinates of the object we are going
# to track
first_frame = True
counter = 0

# initialize the FPS throughput estimator
fps = None
vs = cv2.VideoCapture(args["video"])

# initialize Kalman Filter variables
measurement = np.array((2,1), np.float32)
state = np.array((4,1), np.float32)
prediction = np.zeros((4,1), np.float32)

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	counter+=1

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	if counter%300 == 0:
		first_frame = True

	# check to see if we are currently tracking an object
	if not first_frame:
		# grab the new bounding box coordinates of the object
		all_success = True

		for i, (tracker, kalman) in enumerate(zip(trackers, kalmanfilters)):
			(success, box) = tracker.update(frame)
			all_success = all_success and success

    		# check to see if the tracking was a success
			if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),
    				(0, 255, 0), 2)
                # compute Kalman estimates
				# print(kalmanfilters)
				# prediction = kalman.predict()
				# measurement[0] = x+int(w/2)
				# measurement[1] = y+int(h/2)
				# estimated = kalman.correct(measurement)
				# process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
				# state = np.dot(kalman.transitionMatrix, estimated) + process_noise
				# label = "X-Velocity: {0:.2f}".format(float(state[2]))
				# label1 = "X-Position: {0:.2f}, X-Velocity: {1:.2f}".format(float(state[0]), float(state[2]))
				# print("[INFO] {}".format(label1))
				# cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				#p0 = np.concatenate((p0, np.array([float(x + w/2), float(y+h/2)])))
				#p0.append([float(x + w/2), float(y + h/2)])


				# Calculate Optical Flow
				p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
				# Now update the previous frame and previous points
				#print(p1)
				velocity = (p1-p0)*20*(1/20) #assuming frame rate 30fps
				#print(velocity)
				xvelocity = velocity[0][0]
				yvelocity = velocity[0][1]
				label = "X-Velocity: {0:.2f}".format(float(xvelocity))
				# label1 = "X-Position: {0:.2f}, X-Velocity: {1:.2f}".format(float(state[0]), float(state[2]))
				print("[INFO] {}".format(label))
				cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		old_frame = frame.copy()
		p0 = p1

		# update the FPS counter
		fps.update()
		fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if all_success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# initialise bounding box to track
	if first_frame:
		# select the bounding box of the object we want to track, with 10% tolerance
        # NB initBB's input is (startX, startY, deltaX, deltaY), from selectROI!
		locations, image = detect(frame)
		print(locations)
        # Take first frame and find optimal parameters to track for Optical Flow method
		old_frame = frame
		#p0 = cv2.goodFeaturesToTrack(frame, mask = None, **feature_params)

		trackers = [OPENCV_OBJECT_TRACKERS[args["tracker"]]() for _ in range(len(locations))]
		kalmanfilters = [cv2.KalmanFilter(4, 2, 0) for each in range(len(locations))]

		for i in range(len(locations)):
			kalmanfilters[i].measurementMatrix = np.array([[1,0,0,0],
                                                 [0,1,0,0]],np.float32)

			kalmanfilters[i].transitionMatrix = np.array([[1,0,1,0],
                                                [0,1,0,1],
                                                [0,0,1,0],
                                                [0,0,0,1]],np.float32)

			kalmanfilters[i].processNoiseCov = np.array([[1,0,0,0],
                                               [0,1,0,0],
                                               [0,0,0.1,0],
                                               [0,0,0,0.1]],np.float32)

		for i, (startX, startY, endX, endY) in enumerate(locations):
			initBB = (startX, startY, (endX-startX)*1.1, (endY-startY)*1.1)
			trackers[i].init(frame, initBB)
			p0 = np.array([[(startX + endX)/2, (startY + endY)/2]], dtype=np.float32)

		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		fps = FPS().start()

		#print(initBB)

		first_frame = False

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break



# release the file pointer
vs.release()

# close all windows
cv2.destroyAllWindows()
