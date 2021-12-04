import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

pics=np.load("olivetti_faces.npy")
labels=np.load("olivetti_faces_target.npy")

print("pics: ", pics.shape)
print("labels: ", labels.shape)

# JUST HAVE A LOOK AT THE DATASET
def sneak_peak():
    fig = plt.figure(figsize=(24, 10))
    columns = 10
    rows = 4
    for i in range(1, columns*rows +1):
        img = pics[(10*i-1),:,:]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap = plt.get_cmap('gray'))
        plt.title("person {}".format(i), fontsize=14)
        plt.axis('off')
        
    plt.suptitle("There are 40 distinct persons in the dataset", fontsize=24)
    plt.show()

# sneak_peak() #uncomment to take a look at the =data set 

# JUST RESHAPING pics
Y = labels.reshape(-1,1) 
X=pics.reshape(pics.shape[0], pics.shape[1]*pics.shape[2]) 

print("X shape:",X.shape)
print("Y shape:",Y.shape)

# SPLITITING AT 80:20 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=46)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


list_accuracy = []

Knn = KNeighborsClassifier(n_neighbors = 1) 
Knn.fit(x_train, y_train)
Knn_accuracy = round(Knn.score(x_test, y_test)*100,2)

print("Knn_accuracy is %", Knn_accuracy)


list_accuracy.append(Knn_accuracy)


y_pred = Knn.predict(x_test)
matrixs = confusion_matrix(y_test, y_pred)
print("matrix:")
print(matrixs)
