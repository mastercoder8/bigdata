# - End - to  - End - Tester

import os
import cv2
import sys
import numpy as np
from scipy import misc
from keras import optimizers
from keras.preprocessing import image as image_utils
import json
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

text = 'bbaf2n'
speaker = '1'
video_path = '../resources/Videos1/' + speaker + '/' + text + '.mpg'
align_path = '../resources/Align1/' + speaker + '/' + text + '.align'

cascade_classifier_path = "../Mouth.xml"
frames_path = "../resources/Frames_test/" + speaker + '/' + text + '/'
mouth_path =  "../resources/Mouth_test/" + speaker + '/' + text + '/'
norm_mouth_path = "../resources/Norm_mouth_test/" + speaker + '/' + text + '/'
word_align_path = "../resources/Word_align_test/" + speaker + '/'

final_sentence = []

if not os.path.exists(video_path):
    print('Video path doesnt exist: ' , video_path)
    sys.exit()


##Extract Frames
print("1.1 [Start] Extracting Frames from Video")
frame_count = 1
vc = cv2.VideoCapture(video_path)
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
while rval:
    rval, frame = vc.read()
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    cv2.imwrite(frames_path + str(frame_count) + '.jpg', frame)
    frame_count += 1
    cv2.waitKey(1)
vc.release()
print('1.2 [Done] Frames Extracted from Video')

##Extract Mouth
print("2.1 [Start] Extracting Mouth from Frames")
mouth_cascade = cv2.CascadeClassifier(cascade_classifier_path)
print('2.2 Loading CascadeClassifier')
frame_list = next(os.walk(frames_path))[2]
frame_count = 0
previous_image = None
previous_mouth = None
for frame in frame_list:
    frame_count += 1
    if(frame.endswith('.jpg') and not frame.endswith('75.jpg')):
        imagePath = frames_path + str(frame)
        image = cv2.imread(imagePath)
        if image is None:
            image = previous_image
            if(previous_image is None):
                print('Some issue while reading frame'+ frame)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mouth = mouth_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags = 0|cv2.CASCADE_SCALE_IMAGE
        )
        if len(mouth) == 0 :
           mouth = previous_mouth
        correct_mouth = mouth[0]
        for i in range(1,len(mouth)):
           if mouth[i][1] > mouth[i-1][1]:
               correct_mouth = mouth[i]

        x,y,w,h = correct_mouth

        letter = image[y-20:y-20+h,x:x+w]
        if not os.path.exists(mouth_path):
           os.makedirs(mouth_path)
        letter = misc.imresize(letter, size=(40,40))
        cv2.imwrite(mouth_path + str(frame),letter)
        cv2.destroyAllWindows()
        previous_mouth = mouth
        previous_image = image 

print('2.3 [Done] Mouths Extracted from Frames  ')

## Normalizing Mouth
print('3.1 [Start] Normalizing Mouths')
frame_list = next(os.walk(mouth_path))[2]
for frame in frame_list:
    frame_count += 1
    if(frame.endswith('.jpg') and not frame.endswith('75.jpg')):
        imagePath = mouth_path + str(frame)
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image,(40,40))
        image = np.asarray(image)
        image_min = image[image>0].min()
        image_max = image[image>0].max()
        image = (image - image_min) / (float(image_max - image_min))
        if not os.path.exists(norm_mouth_path):
           os.makedirs(norm_mouth_path)
        f = open(norm_mouth_path+str(frame.split('.', 1)[0]),"wb")
        np.save(f,image)
        cv2.destroyAllWindows()
print('3.2 [Done] Normalized Mouths ')


## Creating Input Train Data - FramesForWord
print('4.1 [Start] Frames to Testing Data')
input_train = []
output_train = []
input_test = []
output_test = []
word_dist = {}
fileptr = open(align_path)
sentence_framesdata = fileptr.read().splitlines()
for word_framedata in sentence_framesdata:
    word_array = np.zeros((40,1600))
    starting_frame = word_framedata.split()[0]
    starting_frame = int(int(starting_frame)/1000)
    starting_frame += 1
    ending_frame = word_framedata.split()[1]
    ending_frame = int(int(ending_frame)/1000)
    word = word_framedata.split()[2]
    final_sentence.append(word)
    if word not in word_dist:
        word_dist[word] = 1
    else:
        word_dist[word] += 1
    word_index = 0

    if not word == "sil":
        while starting_frame <= ending_frame:
            f = open(norm_mouth_path + str(starting_frame), "rb")
            image = np.load(f)
            image = np.resize(image, 1600)
            word_array[word_index] = image
            word_index += 1
            starting_frame += 1

            input_train.append(word_array)
            output_train.append(word)

input_train = np.asarray(input_train)
output_train = np.asarray(output_train)

if not os.path.exists(word_align_path):
    os.makedirs(word_align_path)

print("4.2 [Save] Saving ", word_align_path + 'input_train'+ speaker + '.npz')
f = open(word_align_path + 'input_train'+ str(speaker) + '.npz',"wb")
np.savez_compressed(f, input_train)

print("4.3 [Save] Saving ", word_align_path + 'output_train'+ speaker)
f2 = open(word_align_path + 'output_train'+ str(speaker), "wb")
np.save(f2, output_train)

print("4.4 [Done] Frames for Word created")

## Generate Output
print("5.1 [Start] Generating Validation Data")
path = '../resources/WordAlign_bckp/speaker_output_test1'
word_index = 0
dict = {}
word_list = np.load(path)
for word in word_list:
    if word not in dict:
        if word != "sil":
            dict[word] = word_index
            word_index +=1
output_vector = []
word_list = np.load(word_align_path +'output_train'+str(speaker))

for j in range(len(word_list)):
    cur_vector = [0] * len(dict)
    cur_vector[dict[word_list[j]]] = 1
    output_vector.append(cur_vector)
output_vector = np.asarray(output_vector)
print("5.2 Final Vector Shape: ", output_vector.shape)
np.save(word_align_path + '/final_output_train'+str(speaker),  output_vector)
print('5.3 [Done] Generated Validation Data')

### Reading Model
print('6.1 [Start] Reading Saved Model')
def read_model(weights_filename='../../untrained_weight.h5',
               topo_filename='../../untrained_topo.json'):
    print("Reading Model from "+ weights_filename + " and " + topo_filename)
    print("Please wait, it takes time.")
    with open(topo_filename) as data_file:
        topo = json.load(data_file)
        model = model_from_json(topo)
        model.load_weights(weights_filename)
        print("Finish Reading!")
        return model

model = read_model()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])
print('6.2 Model Compilation Done')
# seed  = 7
# np.random.seed(seed)
input_testdata_path = word_align_path + 'input_train' + str(speaker) + '.npz'
output_testdata_path = word_align_path + 'final_output_train' + str(speaker) + '.npy'

fil = np.load(input_testdata_path)
X_test = fil['arr_0']
y_test = np.load(output_testdata_path)
print("6.3 Running model predict")
X_test_final = list(X_test)

X_test_final = np.array(X_test_final)
prediction = model.predict(X_test_final)
score,acc = model.evaluate(X_test, y_test)
print("Prediction_shape" , prediction.shape)
print("*************************** Printing prediction ***********\n", prediction)
prediction_class = np.argmax(prediction, axis=0)
print("Prediction class", type(prediction_class), prediction_class.shape)
print(prediction_class)
class_names = list(dict.keys())
print("Class Names:", class_names, len(class_names))
print("Prediction class 0:", prediction_class[0])
for i in range(len(prediction_class)):
    print(class_names[prediction_class[i]%len(class_names)], end=' ')
# print(prediction_class[0]+1)
print("\n",prediction[0])
print(final_sentence)
print("Test Score:", score, " Accuracy:", acc)
print("6.4 [Done] Predicting")