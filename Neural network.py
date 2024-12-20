import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("/home/kduplat/Documents/cours ML/TP_Filtrage/train.csv.zip")
np_data = np.array(data)
print(np.shape(np_data))

shuf_data = np_data.copy()
np.random.shuffle(shuf_data)

X_dev = shuf_data[:1000, 1:]
Y_dev = shuf_data[:1000, 0]

X_train = shuf_data[1000:, 1:]
Y_train = shuf_data[1000:, 0]


X_dev_norm = X_dev / 255.
X_train_norm = X_train / 255.

############# Initialisation
def para_init(nb_neurone = 10, nb_pixel = 784):
    W0 = np.random.uniform(-0.5, 0.5, size=(nb_pixel, nb_neurone))
    W1 = np.random.uniform(-0.5, 0.5, size=(nb_neurone, nb_neurone))
    b0 = np.random.uniform(-0.5, 0.5, size=nb_neurone)
    b1 = np.random.uniform(-0.5, 0.5, size=nb_neurone)
    return W0, W1, b0, b1

def ReLU(Z):
    return np.maximum(0,Z)

def fct_softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

def propagation_forward(X, W0, W1, b0, b1):   
    Z0 = np.dot(W0.T, X) + b0
    A0 = ReLU(Z0)
    Z1= np.dot(W1.T,  A0) + b1
    A1 = fct_softmax(Z1)
    
    return Z0, A0, Z1, A1

def der_ReLU(Z):
    return Z > 0

def exp_result(y):
    tab = np.zeros(10)
    tab[y] = 1
    return tab

### This is applied for 1 image (number)
def propagation_backward(X, Y, A1, Z0, A0, W1):
    delta1 = A1 - exp_result(Y)
    dJdW1 = np.outer(A0, delta1.T)
    dJdb1 = delta1
    delta0 = np.dot(W1, delta1) * der_ReLU(Z0)
    dJdW0 = np.outer(X, delta0.T)
    dJdb0 = delta0
    return dJdW0, dJdW1, dJdb0, dJdb1

### dJdW0_L is  for the list of all the images (number)
def update_param(W0, W1, b0, b1, dJdW0_L, dJdW1_L, dJdb0_L, dJdb1_L, lambd):
    W0 = W0 - lambd * np.mean(dJdW0_L, axis=0)
    W1 = W1 - lambd * np.mean(dJdW1_L, axis=0)
    b0 = b0 - lambd * np.mean(dJdb0_L, axis=0)
    b1 = b1 - lambd * np.mean(dJdb1_L, axis=0)
    return W0, W1, b0, b1
    

###########   1.3
def result(A1):
    # print(A1)
    return np.argmax(A1)

def success_rate(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def training(X_train, Y_train ,nb_epoch, W0, W1, b0, b1, lambd,nb_neurone = 10, nb_pixel = 784):
    
    dJdW0_L = np.zeros((len(X_train), nb_pixel, nb_neurone))
    dJdW1_L = np.zeros((len(X_train), nb_neurone, nb_neurone))  
    dJdb0_L = np.zeros((len(X_train), nb_neurone))
    dJdb1_L = np.zeros((len(X_train), nb_neurone))
    dpred = np.zeros(len(X_train))
    
    for i in range(nb_epoch):
        for j in range(len(X_train)):
            Z0, A0, Z1, A1 = propagation_forward(X_train[j], W0, W1, b0, b1)
            dJdW0, dJdW1, dJdb0, dJdb1 = propagation_backward(X_train[j], Y_train[j], A1, Z0, A0, W1)
            
            dJdW0_L[j] = dJdW0
            dJdW1_L[j] = dJdW1
            dJdb0_L[j] = dJdb0
            dJdb1_L[j] = dJdb1
            dpred[j] = result(A1)
            
        W0, W1, b0, b1 = update_param(W0, W1, b0, b1, dJdW0_L, dJdW1_L, dJdb0_L, dJdb1_L, lambd)
        
        if (i + 1) % 10 == 0:
            print("epoch : ", i+1)
            print("accuracy : ", success_rate(Y_train, dpred))
            print("-----------------------------------")
    
    return A1, W0, W1, b0, b1 
    

epoch = 100
lambd = 1
W0, W1, b0, b1 = para_init()

A1, W0, W1, b0, b1  = training(X_train_norm, Y_train, epoch, lambd, W0, W1, b0, b1)

np.savez('data.npz', W0=W0, W1=W1, b0=b0, b1=b1)

def main(W0, W1, b0, b1):
    dpred = np.zeros(len(X_dev_norm))
    
    for j in range(len(X_dev_norm)):
        Z0, A0, Z1, A1 = propagation_forward(X_dev_norm[j], W0, W1, b0, b1)
        dpred[j] = result(A1)
        
    print("Accuracy : ", success_rate(Y_dev, dpred))
    
main()



############Bonus

nb_epoch = 100
lambd =  0.1
A1_2, W0_2, W1_2, b0_2, b1_2 = training(X_train_norm, Y_train , nb_epoch, W0, W1, b0, b1, lambd,nb_neurone = 10, nb_pixel = 784)

main(W0_2, W1_2, b0_2, b1_2)

