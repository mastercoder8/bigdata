import numpy as np
path = "./resources/WordAlign_bckp"
data = np.load(path+"/speaker_final_output_train1.npy")
print(data.shape)
dict = {}
word_index = 0

word_list = np.load(path + '/speaker_output_test2')
for word in word_list:
    if word not in dict:
        if word != "sil":
            dict[word] = word_index
            word_index +=1
print(len(dict))
print(dict)
word_list = np.load(path + '/speaker_output_test2')
for word in word_list:
    if word not in dict:
        if word != "sil":
            dict[word] = word_index
            word_index +=1
print(len(dict))
print(dict)