import cv2
import os
from scipy import misc
previous_mouth = None
previous_image = None

source_path = './resources/Frames/'
destination_path = './resources/Mouth/'
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
speaker_list = next(os.walk(source_path))[1]
for speaker in speaker_list:
    print("Extracting mouth for speaker",speaker)
    videoname_list = next(os.walk(source_path + str(speaker)+'/'))[1]
    for video_name in videoname_list:
        frame_list = next(os.walk(source_path + str(speaker)+'/' + str(video_name) + '/'))[2]
        for frame in frame_list:
            if frame.endswith('.jpg') and not frame.endswith('75.jpg'):
                imagePath = source_path + str(speaker)+'/'+str(video_name)+'/'+str(frame)
                image = cv2.imread(imagePath)
                if image is None:
                    image = previous_image
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                mouth = mouth_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                    flags = 0|cv2.CASCADE_SCALE_IMAGE
                )
                if len(mouth) == 0:
                    mouth = previous_mouth

                correct_mouth = mouth[0]
                for i in range(1,len(mouth)):
                    if mouth[i][1] > mouth[i-1][1]:
                        correct_mouth = mouth[i]

                x,y,w,h = correct_mouth

                letter = image[y-20:y-20+h,x:x+w]
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)

                if not os.path.exists(destination_path + str(speaker)+'/'):
                    os.makedirs(destination_path + str(speaker)+'/')

                if not os.path.exists(destination_path + str(speaker)+'/'+ str(video_name) +'/'):
                    os.makedirs(destination_path + str(speaker)+'/'+ str(video_name) +'/')
                letter = misc.imresize(letter, size=(40,40))
                cv2.imwrite(destination_path + str(speaker)+'/'+ str(video_name) +'/' +str(frame),letter)
                cv2.destroyAllWindows()
                previous_mouth = mouth
                previous_image = image