import cv2
import os
from scipy import misc
import sys
previous_mouth = None
previous_image = None

source_path = './resources/Frames/'
destination_path = './resources/Mouth/'

if not os.path.exists(destination_path):
    os.makedirs(destination_path)

mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
speaker_list = next(os.walk(source_path))[1]
print("Total Speakers:",len(speaker_list),end='')
for speaker in speaker_list:
    print("\nExtracting mouth for speaker",speaker,end=' [ ')
    videoname_list = next(os.walk(source_path + str(speaker)+'/'))[1]
    if not os.path.exists(destination_path + str(speaker)+'/'):
        os.makedirs(destination_path + str(speaker)+'/')
    i = 0
    count = 0
    video_count = len(videoname_list)
    for video_name in videoname_list:
        if(i%50==0):
            print(str(int(i/10))+"%"+'='*int(i/50)+'>'+'-'*int(20-i/50),end="]"+'\b'*25)
            sys.stdout.flush()
        i += 1
        if not os.path.exists(destination_path + str(speaker)+'/'+ str(video_name) +'/'):
            os.makedirs(destination_path + str(speaker)+'/'+ str(video_name) +'/')
        frame_list = next(os.walk(source_path + str(speaker)+'/' + str(video_name) + '/'))[2]
        frame_count = len(next(os.walk(destination_path + str(speaker)+'/' + str(video_name) + '/'))[2])
        if(frame_count >= 74): # Already has Mouths extracted
            count += frame_count
            continue
        else:
            for frame in frame_list:
                frame_count = frame_count+1
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

                    letter = misc.imresize(letter, size=(40,40))
                    cv2.imwrite(destination_path + str(speaker)+'/'+ str(video_name) +'/' +str(frame),letter)
                    cv2.destroyAllWindows()
                    previous_mouth = mouth
                    previous_image = image
        count += frame_count
    print("Videos:",video_count,"Frames:",count,end=']')