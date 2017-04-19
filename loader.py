import os
import cv2
import time
import sys
source_path = './resources/Videos/'
destination_path = './resources/Frames/'

speaker_list = next(os.walk(source_path))[1]
print("Total Speakers:",len(speaker_list),end='')
for speaker in speaker_list:
    print("\nSpeaker:"+speaker,end=' [ ')
    i=0
    if not os.path.exists(destination_path + str(speaker)+'/'):
        os.makedirs(destination_path + str(speaker)+'/')
    count = 0
    video_count = len(next(os.walk(destination_path + str(speaker)+'/'))[1])
    for filename in os.listdir(source_path + str(speaker)+'/'):
        if filename.endswith('.mpg'):
            i += 1
            if(i%50==0):
                print(str(int(i/10))+"%"+'='*int(i/50)+'>'+'-'*int(20-i/50),end="]"+'\b'*25)
                sys.stdout.flush()
            frame_count = 1
            vc = cv2.VideoCapture(source_path + str(speaker)+'/' + filename)
            if vc.isOpened():
                rval , frame = vc.read()
            else:
                rval = False

            if not os.path.exists(destination_path + str(speaker)+'/'+ str(filename) +'/'):
                os.makedirs(destination_path + str(speaker)+'/'+ str(filename) +'/')
            frame_count = len(next(os.walk(destination_path + str(speaker)+'/'+ str(filename) +'/'))[2])
            if(frame_count==75): #skip reading
                count += frame_count
                continue
            else :  
                while rval:
                    rval, frame = vc.read()
                    cv2.imwrite(destination_path + str(speaker)+'/'+ str(filename) +'/' +str(frame_count)+'.jpg',frame)
                    frame_count = frame_count + 1
                    cv2.waitKey(1)
            vc.release()
            count += frame_count
    print("\bVideos:",video_count,"Frames:",count,end=']')