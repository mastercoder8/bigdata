import numpy as np
from keras.utils import np_utils, generic_utils





def Data(max_seqlen=30, image_size=1600, data_num=1000):
    return np.random.rand(data_num, max_seqlen, image_size)

def Label(data_num=1000,label_size=53):
    label_array = range(label_size)
    return np_utils.to_categorical(np.random.choice(label_array, size=data_num))


def Train(data_num=6000,label_size=53):
    return Data(data_num=data_num), Label(data_num=data_num,label_size=label_size)


def Test(data_num=1000,label_size=53):
    return Data(data_num=data_num), Label(data_num=data_num,label_size=label_size)


def Val(data_num=1000,label_size=53):
    return Data(data_num=data_num), Label(data_num=data_num,label_size=label_size)

