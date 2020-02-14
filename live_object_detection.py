# USAGE
# python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# import the necessary packages
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the list of object trackers and corresponding class
# labels
labels = []

# start the frames per second throughput estimator
#fps = FPS().start()

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)
unknown_cnt = 0
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
	if True:
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# pass the blob through the network and obtain the detections
		# and predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]
				print(CLASSES[idx])
				# if the class label is not a person, ignore it
				if (CLASSES[idx] == "person" ) or (CLASSES[idx] == "car" ):
					
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# grab the corresponding class label for the detection
					# and draw the bounding box
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 255, 0), 2)
					cv2.putText(frame, label, (startX, startY - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
					if(CLASSES[idx]=="person"):
						unknown_cnt+=1
						if(unknown_cnt>50):
							print("Person Detected")
							time.sleep(30)
							unknown_cnt=0
					elif(CLASSES[idx]=="car"):
						unknown_cnt+=1
						if(unknown_cnt>50):
							print("Car Detected")
							time.sleep(30)
							unknown_cnt=0
		
	

	# check to see if we are supposed to display the output frame to
	# the screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#fps.stop()
	#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

