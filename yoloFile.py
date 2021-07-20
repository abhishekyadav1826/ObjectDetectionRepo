
"""
	     GRIPJULY21

	Name -----ABHISHEK YADAV
	TASK 1-----OBJECT DETECTION



"""
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
agp= argparse.ArgumentParser()
agp.add_argument("-i", "--image", required=True,
	help="path to input image")
agp.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections, IoU threshold")
agp.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(agp.parse_args())

namesPath = 'yolo-coco\\coco.names'
NAMES = open(namesPath).read().strip().split("\n")


COLORS = np.random.randint(0, 255, size=(len(NAMES), 3),
	dtype="uint8")

weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(layerNames)

bboxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([W, H, W, H])
			(BX, BY, wd, ht) = box.astype("int")
			x = int(BX - (wd / 2))   #top corner of bounding box
			y = int(BY - (ht / 2))   #left corner of the bounding box
			bboxes.append([x, y, int(wd), int(ht)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
indxs = cv2.dnn.NMSBoxes(bboxes, confidences, args["confidence"],
	args["threshold"])

if len(indxs) > 0:
	for i in indxs.flatten():
		(x, y) = (bboxes[i][0], bboxes[i][1])
		(w, h) = (bboxes[i][2], bboxes[i][3])
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(NAMES[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN,
			0.5, color, 1)
cv2.imshow("Image", image)
cv2.waitKey(0)
