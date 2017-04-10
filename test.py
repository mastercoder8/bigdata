import lipreadtrain
import random_data
import real_data

data = random_data
label_size = 53



X_train, y_train = data.Train()
X_test,y_test = data.Test(label_size=label_size)

net = lipreadtrain.build_network(dict_size=label_size)

lipreadtrain.train(model=net,
                   X_train=X_train, y_train=y_train,
                   X_test=X_test, y_test=y_test)
