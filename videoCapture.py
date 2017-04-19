import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy
import time
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
	MOUTH_POINTS,
]
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass
def draw_convex_hull(im, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)
 
def get_landmarks(im):
	rects = detector(im, 1)
	
	if len(rects) > 2:
		raise TooManyFaces
	if len(rects) == 0:
		raise NoFaces

	return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def get_face_mask(im, landmarks):
	im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

	for group in OVERLAY_POINTS:
		draw_convex_hull(im,
						 landmarks[group],
						 color=1)

	im = numpy.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

	return im

def read_im_and_landmarks(fname):
	im = cv2.imread(fname, )
	im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
						 im.shape[0] * SCALE_FACTOR))
	s = get_landmarks(im)

	return im, s

	
def show_webcam(mirror=False):
	win = dlib.image_window()
	cam = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vw = cv2.VideoWriter('output.avi',fourcc, 25.0, (640,480))
	i = 0
	while cam.isOpened():
		ret_val, img = cam.read()
		# print(ret_val, img)
		if mirror: 
			img = cv2.flip(img, 1)
		# cv2.imshow('my webcam', img)
		# if cv2.waitKey(1) == 27: 
		# 	break  # esc to quit
		# win.clear_overlay()
		win.set_image(img)
		im = cv2.resize(img, (img.shape[1] * SCALE_FACTOR,img.shape[0] * SCALE_FACTOR))
		s = get_landmarks(im)
		dets = detector(img, 1)
		out = img
		print("Number of faces detected: {}".format(len(dets)))
		for k, d in enumerate(dets):
			print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
			shape = predictor(img,d)
			print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
		cv2.rectangle(out, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
		cv2.putText(out, "Face marks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
		win.add_overlay(shape)
		# cv2.imshow('my webcam', img)
		# if cv2.waitKey(1) == 27: 
			# break  # esc to quit
		win.add_overlay(dets)
		# dlib.hit_enter_to_continue()
		cv2.waitKey(25)
		print("Saving File:",'Frame'+str(i)+'-output-dlib68.jpg')
		cv2.imwrite('./resources/capture/Frame'+str(i)+'-output-dlib68.jpg', img)
		vw.write(img)
		i += 1		
	# cv2.destroyAllWindows()
	vw.release()
	cam.release()

def main():
	show_webcam(mirror=True)

if __name__ == '__main__':
	main()