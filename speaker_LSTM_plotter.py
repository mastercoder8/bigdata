from __future__ import print_function
from builtins import str
# from builtins import range
import lipreadtrain
import gc
import real_data
import numpy as np
# import resource
from copy import copy
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt

data = real_data

net = lipreadtrain.build_network(dict_size=52,
                                lr=0.0002,
                                max_seqlen=40,
                                image_size=(40,40),
                                optimizer='rmsprop',load_cache=False,save_weight_to='./Final_weights/untrained_weight.h5',save_topo_to='./Final_weights/untrained_topo.json')

test_acc = []
X_test_final = []
y_test_final = []
input_traindata_path = "./resources/WordAlign_bckp/speaker_input_train"
output_traindata_path = "./resources/WordAlign_bckp/speaker_final_output_train"
input_testdata_path = "./resources/WordAlign_bckp/speaker_input_test"
output_testdata_path = "./resources/WordAlign_bckp/speaker_final_output_test"


for speaker_id in range(6,7): #range(1,33)
	# for j in range(0,12):
	fil = np.load(input_traindata_path + str(speaker_id)+'.npz')#+'_'+str(j)+".npz")
	X_train = fil['arr_0']
	y_train = np.load(output_traindata_path + str(speaker_id)+".npy")
	# y_train = y_train[j*500:(j+1)*500]
	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	fil2 = np.load(input_testdata_path+ str(speaker_id)+'.npz')
	X_test = fil2['arr_0']
	y_test = np.load(output_testdata_path + str(speaker_id) + ".npy")
	#print(">>>>>>>>>>>>",X_train.shape,y_train.shape)
	print("Weights Loaded")
	history = net.fit(X_train, y_train, batch_size=100, epochs=10, validation_split=0.3)#, show_accuracy=True)
	print(history.history.keys())
	print("Accuracy plot running")
	##Accuracy plot
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
	# summarize history for loss
	print("Loss Plot running")
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	X_test_final = X_test_final + list(X_test)
	y_test_final = y_test_final + list(y_test)
	del X_train
	del y_train
	fil.close()
	gc.collect()
X_test_final = np.array(X_test_final).astype(float)
y_test_final = np.array(y_test_final).astype(float)
#print(">>>>>>>>>>>>",X_test_final.shape,y_test_final.shape)
score,acc = net.evaluate(X_test_final,y_test_final)#,show_accuracy=True)
print("Test accuracy ",acc," Test score ",score)
