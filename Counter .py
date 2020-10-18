#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import argparse
import imutils
import time
import cv2
import os

#Put own path to file here 
video = "castro.mov"
conf = 0.5
threshold = 0.1

classname = []
list_of_vehicles = ["car","truck"]
def get_vehicle_count(boxes, class_names):
	total_vehicle_count = 0 # total vechiles present in the image
	dict_vehicle_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		class_name = class_names[i]
		# print(i,".",class_name)
		if(class_name in list_of_vehicles):
			total_vehicle_count += 1
			dict_vehicle_count[class_name] = dict_vehicle_count.get(class_name,0) + 1

	return total_vehicle_count, dict_vehicle_count


ped = ["person"]
def get_ped_count(boxes, class_names):
	total_ped_count = 0 # total vechiles present in the image
	dict_ped_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		class_name = class_names[i]
		# print(i,".",class_name)
		if(class_name in ped):
			total_ped_count += 1
			dict_ped_count[class_name] = dict_ped_count.get(class_name,0) + 1

	return total_ped_count, dict_ped_count

#Own Path here too
# load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

#Own Path here too
# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(video)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() 		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream


framecount = 0
totalvech = 0
totalped = 0
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > conf:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				classname.append(LABELS[classID])

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,threshold)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None

	total_vehicles,vech = get_vehicle_count(boxes, classname)
	print("Total vechicles in frame  " + str(framecount) +" : ",total_vehicles)
	total_ped,ped = get_ped_count(boxes,classname) 
	print("Total pedestrians in frame  " + str(framecount) +" : ",total_ped)
	font = cv2.FONT_HERSHEY_SIMPLEX    
	cv2.putText(frame,"Pedestrian Count: " + str(total_ped),(50, 50),font, 1,(0, 255, 255),  2, cv2.LINE_4)  
	cv2.putText(frame,"Vehicle Count: " + str(total_vehicles),(20, 20),font, 1,(0, 220, 220),  2, cv2.LINE_4) 
	framecount = int(framecount) + 1
	totalvech += total_vehicles
	totalped += total_ped
	#print(totalped)  
	if writer is None:
		# initialize our video writer. Put file type in for *""
		fourcc = cv2.VideoWriter_fourcc(*"MOVV")
		writer = cv2.VideoWriter("tester.mov", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			workTimes = elap * total
			workHours = workTimes / 3600 
			print("[INFO] every one frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish process: {:.2f} minutes".format(
				workTimes / 60))

	# write the output frame to disk
	writer.write(frame)

print("Done!")
writer.release()
vs.release()


# In[ ]:





# In[ ]:




