import socket
import binascii

import picamera
from picamera.array import PiRGBArray
import io

import argparse
import signal
import sys
import time
import logging
import base64
import cv2
import config
import os
import datetime

import numpy as np

# MQTT
broker = '192.168.10.1' # IP or DNS of Heimdall-Backend

# Code from https://github.com/jrosebr1/imutils/blob/1f111c0edaae3d5b4f57f73baf2ea7ac31cffafd/imutils/convenience.py#L65
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()



def filenames():
    frame = 0
    while True:
        if frame != 0:
            filename = 'image%02d.jpg' % (frame-1)
            post_image(filename)
            os.remove(filename)
        yield 'image%02d.jpg' % frame
        frame = frame+1

        if frame > 100:
            break

def post_image_by_filename(filename):
    print('###############')
    print('Taking photo: ' + filename)

    with open(filename, "rb") as image_file:
        #data = base64.b64encode(image_file.read())
        data = image_file.read()

    post_image(data)


def post_image_opencv2(data):
    retval, buffer = cv2.imencode('.jpg', data)
    #imgBase64 = base64.b64encode(buffer)
    #post_image(imgBase64)
    post_image(buffer)


def post_image(data):
    data = binascii.hexlify(data)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((broker, 9000))
        s.send(data)
        s.close()
    except TimeoutError as e:
        print(e)
    except ConnectionRefusedError as e:
        print(e)

#    faces = detect_faces(image)
#    if len(faces) > 0:
#        print("Found {} face(s)".format(len(faces)))
#        #for (x,y,w,h) in faces:
#        #    cv2.rectangle(image, (x, y), (x+w, y+h), 255)
#        #    cv2.imwrite('detect.jpg', image)
#
#        client.publish('camera', data, mqttQos, mqttRetained)
#        print('Image published')
#    else:
#        print('No Face Detected :(')


resolution = (800, 600)
#resolution = (1280,1024)
#resolution = (1640, 1232)
delta_thresh = 5
min_area = 4000
blurry_threshold = 100

avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
motionStopDelayConst = 10 # number of frames before stop
motionStopDelay = 0
delay = 0.0 # 200ms

try:
    with picamera.PiCamera() as camera:
        rawCapture = PiRGBArray(camera, size=resolution)

        #camera.exposure_mode = 'sports'
        camera.resolution = resolution
        #camera.vflip=True
        #camera.hflip=True
        #camera.framerate = 2
        camera.framerate = 1
        #camera.start_preview()
        time.sleep(2) #warm-up

        # capture frames from the camera
        for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            #print("Grabbed frame!")
            motionCounter = 0

            #print(camera.shutter_speed)
            #print(camera.exposure_speed)


            # grab the raw NumPy array representing the image and initialize
            # the timestamp and occupied/unoccupied text
            frame = f.array
            origFrame = frame
            timestamp = datetime.datetime.now()
            text = "Unoccupied"

            # resize the frame, convert it to grayscale, and blur it
            frame = resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the average frame is None, initialize it
            if avg is None:
                print("[INFO] starting background model...")
                avg = gray.copy().astype("float")
                rawCapture.truncate(0)
                continue

            # accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))


            # threshold the delta image, dilate the thresholded image to fill
            # in holes, then find contours on thresholded image
            thresh = cv2.threshold(frameDelta, delta_thresh, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1]

            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            totalContArea = 0
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                cntArea = cv2.contourArea(c)
                if cntArea < min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(frame, "{}".format(cntArea), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(thresh_color, "{}".format(cntArea), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                totalContArea  += cntArea
                motionCounter += 1

                motionStopDelay = motionStopDelayConst

            # draw the text and timestamp on the frame
            #ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            #cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
            #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #        0.35, (0, 0, 255), 1)

            # check to see if the frames should be displayed to screen
            if False:
		        # display the security feed
                # thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                merged = np.concatenate((frame, thresh_color), 1)
                cv2.imshow("Security Feed 2", merged)
                #cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break

            motionStopDelay -= 1

            #if motionCounter > 0:
            if motionStopDelay > 0:
                print(timestamp.strftime("%d.%m.%Y %H:%M:%S") + " - Motion detected - sending image - stopping in: ", motionStopDelay)
                post_image_opencv2(origFrame)
                time.sleep(delay)
            #print("Motion Area: {}".format(totalContArea))

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
except KeyboardInterrupt:
    print("Stop!")
